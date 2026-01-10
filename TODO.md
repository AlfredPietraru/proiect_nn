# TODO – SUP → EMA → KDD Distillation Pipeline

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision
sudo apt-get update && sudo apt-get install -y build-essential python3-dev
pip3 install -U pycocotools dataset_tools
pip install -r requirements.txt
```

This document specifies **all experiments**, their **exact execution order**, and **allowed interactions** for the SUP → EMA → KDD study, including cross-dataset transfer.  
The goal is to isolate **learning mechanism effects** (EMA vs KDD) from **architecture** and **dataset** effects.

==> YOLO / RESNET50-GRADCAM++ / FASTER-RCNN-RESNET50 [TO WORK FOR STAGE PIPELINES MUST BE UPDATED]
==> MAKE SAME RUNS FOR EACH MODEL FOR SUP-BURN-IN ON EACH DATASET VOC=>DRONES SETS
==> MAKE SAME RUNS FOR EACH MODEL FOR UBIASED-TEACHER ON EACH DATASET VOC=>DRONES SETS
==> KDD BETWEEN 2 MODELS

---

## CORE RULES (DO NOT VIOLATE)

- [ ] **Never mix datasets** unless explicitly performing **TRANSFER**
- [ ] All stages operate on **ONE dataset at a time**
- [ ] Every **KDD result is compared only against its EMA baseline**
- [ ] Training order is **fixed and irreversible**:

→ SUP (burn-in)
→ SEMI-SUP (weak–strong, teacher–student)
→ EMA (teacher stabilization)
→ KDD (KL distillation)

- [ ] No stage is skipped, merged, or reordered
- [ ] Transfer is **forbidden** until all per-dataset baselines are complete

---

## GLOBAL TRAINING TEMPLATE (APPLIES EVERYWHERE)

For **any (Model, Dataset)** pair, run the following **four logical stages**:

1. **SUP**  
   Supervised burn-in on labeled data only  
   → establishes a dataset-specific, architecture-specific baseline

2. **SEMI-SUP (Unbiased Teacher)**  
   Weak–strong augmentation + teacher–student learning  
   → enables semi-supervised learning via pseudo-labels

3. **EMA (Teacher Stabilization)**  
   Teacher updated as EMA of student parameters  
   → stabilizes pseudo-label quality and removes confirmation bias

4. **KDD (EMA → KL Transition)**  
   Hard pseudo-labels replaced by KL divergence  
   → transfers *distributional* knowledge from EMA teacher

> NOTE  
> Stages (2) and (3) run **together in practice**,  
> but are **conceptually separated** for clarity and analysis.

---

## BLOCK A – VOC DATASET (NO TRANSFER)

**Purpose**
- Validate semi-supervised learning behavior
- Establish EMA baselines
- Measure KDD improvement over EMA
- Ensure gains are **model-dependent**, not dataset artifacts

VOC is treated as a **closed world** in this block.

---

### VOC + Faster R-CNN ResNet50-FPN

- [ ] **SUP on VOC**
  - [ ] Labeled VOC images only
  - [ ] Standard detection loss (classification + regression)

- [ ] **SEMI-SUP on VOC (Unbiased Teacher)**
  - [ ] Weak–strong augmentation
  - [ ] Teacher generates pseudo-labels on weak view
  - [ ] Student learns on strong view

- [ ] **EMA on VOC**
  - [ ] Teacher updated via EMA(student)
  - [ ] Teacher has no gradients (inference only)

- [ ] **KDD on VOC (same architecture)**
  - [ ] Teacher = frozen EMA model
  - [ ] KL divergence on unlabeled VOC
  - [ ] Confidence-weighted distillation

**Metrics**
- [ ] mAP
- [ ] Pseudo-label confidence distribution
- [ ] EMA vs KDD comparison

---

### VOC + ResNet50 + CAM Head

- [ ] **SUP on VOC**
- [ ] **SEMI-SUP**
  - [ ] Weak–strong classification consistency
  - [ ] CAM-guided pseudo-label supervision
- [ ] **EMA stabilization**
- [ ] **KDD**
  - [ ] KL on class probabilities
  - [ ] Optional KL on CAM activations

**Metrics**
- [ ] Classification accuracy
- [ ] CAM → bbox localization consistency
- [ ] EMA vs KDD comparison

---

### VOC + YOLO11n

- [ ] **SUP on VOC**
- [ ] **SEMI-SUP**
  - [ ] Weak–strong augmentation
  - [ ] Anchor-free pseudo-label generation
- [ ] **EMA stabilization**
- [ ] **KDD**
  - [ ] KL on objectness and class distributions

**Metrics**
- [ ] mAP
- [ ] Pseudo-label density
- [ ] Confidence distribution
- [ ] EMA vs KDD comparison

---

## BLOCK B – DRONE DATASETS (NO TRANSFER)

**IMPORTANT**
- [ ] VOC is **NOT used**
- [ ] Each dataset is treated **independently**

**Purpose**
- Verify SUP → EMA → KDD works per dataset
- Remove dataset-difficulty and class-distribution confounders

---

### VisDrone + Same Architecture

- [ ] SUP on VisDrone
- [ ] SEMI-SUP (weak–strong)
- [ ] EMA stabilization
- [ ] KDD on VisDrone

**Metrics**
- [ ] mAP
- [ ] EMA vs KDD comparison

---

### UAVDT + Same Architecture

- [ ] SUP on UAVDT
- [ ] SEMI-SUP
- [ ] EMA
- [ ] KDD

**Metrics**
- [ ] mAP
- [ ] EMA vs KDD comparison

---

### AU-AIR + Same Architecture

- [ ] SUP on AU-AIR
- [ ] SEMI-SUP
- [ ] EMA
- [ ] KDD

**Metrics**
- [ ] mAP
- [ ] EMA vs KDD comparison

---

## BLOCK C – TRANSFER EXPERIMENTS (ONLY AFTER A + B)

**IMPORTANT**
- [ ] Transfer starts **only after all per-dataset baselines**
- [ ] Datasets are mixed **only in this block**

---

### VOC → Drone Datasets (Cross-Dataset Transfer)

**Teacher**
- [ ] EMA or KDD model trained on VOC

**Student**
- [ ] VisDrone
- [ ] UAVDT
- [ ] AU-AIR

**Data**
- [ ] Unlabeled drone images only

**Loss**
- [ ] KL divergence
- [ ] Weak → strong consistency
- [ ] Confidence-weighted
- [ ] **Only overlapping classes**
- [ ] Optional feature-level KDD (cross-architecture only)

**Purpose**
- Answer the question:

> *Does knowledge learned on VOC transfer to drone datasets?*

---

## ONE-LINE PIPELINE SUMMARY

For each dataset and each model, we train supervised, then semi-supervised with EMA, then replace pseudo-labels with KL distillation.  
Only after establishing EMA and KDD baselines per dataset do we perform cross-dataset transfer.
