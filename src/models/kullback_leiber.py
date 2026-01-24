from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from bbox.box_ops import box_iou

EPS: float = 1e-6

def softmax_temp(logits: Tensor, tau: float) -> Tensor:
    # logits: (..., C) unnormalized scores
    return F.softmax(logits / tau, dim=-1)


def kl_teacher_student(p_t: Tensor, p_s: Tensor) -> Tensor:
    # p_t, p_s: (..., C) probabilities
    return F.kl_div(p_s.clamp_min(EPS).log(), p_t.clamp_min(EPS), reduction="none").sum(dim=-1)


def confidence_from_probs(p_t: Tensor) -> Tensor:
    # p_t: (N,C) or (...,C)
    return p_t.max(dim=-1).values


def confidence_weight(c: Tensor, gamma: float) -> Tensor:
    # w = clip((c - gamma)/(1-gamma), 0, 1)
    denom = max(EPS, 1.0 - float(gamma))
    w = (c - float(gamma)) / denom
    return w.clamp_(0.0, 1.0)


def smooth_distribution(p: Tensor) -> Tensor:
    # q = (1-eps)*p + eps*(1/k)
    k = p.shape[-1]  # number of classes
    return (1.0 - float(EPS)) * p + float(EPS) * (1.0 / float(k))


class ClassProjector:
    def __init__(self, teacher_to_student: dict[int, int], ks: int) -> None:
        self.teacher_to_student = dict(teacher_to_student)
        self.ks = int(ks)

        if self.ks <= 0:
            raise ValueError("ks must be > 0")

        t_idx, s_idx = [], []
        for t, s in sorted(self.teacher_to_student.items(), key=lambda x: x[1]):
            t_idx.append(int(t))
            s_idx.append(int(s))
        self._t_idx = Tensor(t_idx, dtype=torch.long)
        self._s_idx = Tensor(s_idx, dtype=torch.long)

    def to(self, device: torch.device) -> ClassProjector:
        """Move internal tensors to device."""
        self._t_idx = self._t_idx.to(device)
        self._s_idx = self._s_idx.to(device)
        return self

    def project_probs(self, p_t: Tensor) -> Tensor:
        """Project teacher probabilities to student class space."""
        out = p_t.new_zeros((*p_t.shape[:-1], self.ks))
        if self._t_idx.numel() == 0:
            return out

        out.index_copy_(-1, self._s_idx, p_t.index_select(-1, self._t_idx))
        z = out.sum(dim=-1, keepdim=True).clamp_min(EPS)
        return out / z


class WeakStrongKDD(nn.Module):
    def __init__(self,  tau: float = 2.0,  gamma: float = 0.7) -> None:
        super().__init__()
        self.tau = tau
        self.gamma = gamma

    def forward(self, teacher_logits_w: Tensor, student_logits_s: Tensor, weight: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute the Kullback-Leibler divergence loss between teacher and student logits with temperature scaling and confidence weighting."""
        p_t = softmax_temp(teacher_logits_w, self.tau)
        p_s = softmax_temp(student_logits_s, self.tau)

        p_t = smooth_distribution(p_t)

        kl = kl_teacher_student(p_t, p_s)  # (N,)
        c = confidence_from_probs(p_t)     # (N,)

        if weight is not None:
            weight = weight.to(kl.device, dtype=kl.dtype)
        else:
            weight = confidence_weight(c, self.gamma)

        loss = (weight * (self.tau * self.tau) * kl).mean()
        return loss, c.detach(), weight.detach()


class CrossDatasetKDD(nn.Module):
    def __init__(self, projector: ClassProjector, tau: float = 2.0, gamma: float = 0.7) -> None:
        super().__init__()
        self.projector = projector
        self.tau = tau
        self.gamma = gamma

    def forward(self, teacher_logits_w: Tensor, student_logits_s: Tensor, weight: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute the Kullback-Leibler divergence loss between teacher and student logits with class projection, temperature scaling, and confidence weighting."""
        p_t = softmax_temp(teacher_logits_w, self.tau)
        p_t = self.projector.project_probs(p_t)
        p_t = smooth_distribution(p_t)

        if p_t.sum(dim=-1).max().item() == 0.0:
            # no overlap -> skip safely
            z = teacher_logits_w.sum() * 0.0
            return z, p_t.new_zeros((p_t.shape[0],)), p_t.new_zeros((p_t.shape[0],))

        p_s = softmax_temp(student_logits_s, self.tau)

        kl = kl_teacher_student(p_t, p_s)
        c = confidence_from_probs(p_t)

        if weight is not None:
            weight = weight.to(kl.device, dtype=kl.dtype)
        else:
            weight = confidence_weight(c, self.gamma)

        loss = (weight * (self.tau * self.tau) * kl).mean()
        return loss, c.detach(), weight.detach()


class FeatureKDD(nn.Module):
    def __init__(self, proj: nn.Module, beta: float = 1.0) -> None:
        super().__init__()
        self.proj = proj
        self.beta = float(beta)

    def forward(self, f_t: Tensor, f_s: Tensor) -> Tensor:
        """Compute feature-based knowledge distillation loss between teacher and student features."""
        # f_t, f_s: (N,C,H,W) or (N,C)
        gs = self.proj(f_s)
        ft = f_t

        gs = gs.flatten(1)
        ft = ft.flatten(1)

        gs = gs / gs.norm(dim=1, keepdim=True).clamp_min(1e-12)
        ft = ft / ft.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return self.beta * (gs - ft).pow(2).sum(dim=1).mean()


class BoxMatchKDD(nn.Module):
    def __init__(self, tau: float = 2.0, gamma: float = 0.7, iou_thr: float = 0.5, box_l1: float = 0.0) -> None:
        super().__init__()
        self.tau = tau
        self.gamma = gamma
        self.iou_thr = iou_thr
        self.box_l1 = box_l1

    def forward(self, t_boxes: Tensor, t_logits: Tensor, t_valid: Tensor, s_boxes: Tensor, s_logits: Tensor, s_valid: Tensor) -> Tensor:
        """Compute box-matching knowledge distillation loss between teacher and student boxes and logits."""
        N = t_boxes.shape[0]
        loss_sum = t_boxes.new_zeros(())
        denom = 0

        for i, _ in enumerate(range(N)):
            tb, tl = t_boxes[i][t_valid[i]], t_logits[i][t_valid[i]]
            sb, sl = s_boxes[i][s_valid[i]], s_logits[i][s_valid[i]]

            # no boxes -> skip
            if tb.numel() == 0 or sb.numel() == 0:
                continue

            ious = box_iou(tb, sb)  # (T,S)
            best_iou, best_j = ious.max(dim=1)
            keep = best_iou >= self.iou_thr
            if keep.sum().item() == 0:
                continue

            tl, sl = tl[keep], sl[best_j[keep]]
            tb, sb = tb[keep], sb[best_j[keep]]

            p_t = softmax_temp(tl, self.tau)
            p_s = softmax_temp(sl, self.tau)

            kl = kl_teacher_student(p_t, p_s)
            c = confidence_from_probs(p_t)
            w = confidence_weight(c, self.gamma)

            loss = (w * (self.tau * self.tau) * kl).mean()

            if self.box_l1 > 0:
                loss = loss + self.box_l1 * (sb - tb).abs().mean()

            loss_sum = loss_sum + loss
            denom += 1

        if denom == 0:
            return loss_sum
        return loss_sum / float(denom)
