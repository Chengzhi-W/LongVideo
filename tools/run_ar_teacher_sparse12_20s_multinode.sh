#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export NNODES="${NNODES:-2}"
exec bash "$BASE/tools/run_ar_teacher_sparse12_20s.sh" "$@"
