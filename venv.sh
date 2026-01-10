#!/bin/bash
set -euo pipefail

# Step 0 (Local): Setup Python environment for local debugging/testing before using the cluster.
# The PDF recommends not relying on heavy remote FS sync; local venv is the simplest local workflow.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${REPO_DIR:-${SCRIPT_DIR}}"

VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
REQ_FILE="${REQ_FILE:-${REPO_DIR}/requirements.txt}"

echo "[venv] Repo: ${REPO_DIR}"
echo "[venv] Venv: ${VENV_DIR}"
echo "[venv] Requirements: ${REQ_FILE}"

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "ERROR: requirements.txt not found at: ${REQ_FILE}"
  exit 1
fi

python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${REQ_FILE}"

echo "[venv] OK. Activate with:"
echo "  source \"${VENV_DIR}/bin/activate\""
