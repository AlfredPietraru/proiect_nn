#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"

echo "[run] Repo: ${REPO_DIR}"
source "${REPO_DIR}/.venv/bin/activate"
python3 "${REPO_DIR}/main.py" "$@"
echo "[run] Done."
