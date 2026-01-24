#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"

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
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${REQ_FILE}"

echo "[venv] OK. Activate with:"
echo "  source \"${VENV_DIR}/bin/activate\""
echo "[venv] Done."
