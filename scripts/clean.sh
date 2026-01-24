#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"

echo "[clean] Repo: ${REPO_DIR}"

rm -rf "${REPO_DIR}/__pycache__" || true
find "${REPO_DIR}" -type d -name "__pycache__" -exec rm -rf {} + || true

echo "[clean] Done."
