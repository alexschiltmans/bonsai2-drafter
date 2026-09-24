#!/bin/bash
# Lint, types and tests in one command. mypy and the GPU tiers need the pinned dspark
# environment (envs/dspark/), and the GPU tiers need Apple Silicon. They use $PY if it is set,
# else ./.venv/bin/python if it exists (scripts/serve-bonsai2.sh builds the environment there),
# else $HOME/.venv-dspark/bin/python.
#
#     bench/check.sh            ruff, mypy, the tests that need no GPU
#     bench/check.sh --gpu      plus the GPU tests
#     bench/check.sh --models   plus the model-backed checks (8.6 GB pack + 3.9 GB drafter)
#
# Not registered here: test_dflash_prequantized.py --source/--artifact, which needs an
# exported artifact. bench/drafter/README.md has the command.
set -u
cd "$(dirname "$0")/.." || exit 1
if [ -z "${PY:-}" ]; then
  if [ -x .venv/bin/python ]; then PY=$PWD/.venv/bin/python; else PY=$HOME/.venv-dspark/bin/python; fi
fi
rc=0
run() { printf '%-44s' "$1"; shift; out=$("$@" 2>&1); r=$?; if [ $r = 0 ]; then echo ok; else echo FAIL; echo "$out" | tail -15; rc=1; fi; }
run "ruff"                        uvx ruff check .
run "mypy --strict"               uvx mypy --python-executable "$PY"
run "shellcheck"                  uvx --from shellcheck-py shellcheck -S warning bench/check.sh scripts/serve-bonsai2.sh
run "test_served_accept_analysis" python3 bench/tests/test_served_accept_analysis.py
run "test_report_schema"          python3 bench/tests/test_report_schema.py
run "test_dflash_prequantized"    python3 bench/tests/test_dflash_prequantized.py
run "test_bench5"                 python3 bench/tests/test_bench5.py
run "test_pool"                   python3 bench/tests/test_pool.py
run "rename-codebooks --self-test" python3 scripts/rename-codebooks.py --self-test
run "dflash-mlx adapter --self-test" python3 bench/adapters/dflash_mlx_bonsai2.py --self-test
if [ "${1:-}" = --gpu ] || [ "${1:-}" = --models ]; then
  run "test_kv_group_patch"       "$PY" bench/tests/test_kv_group_patch.py
  run "test_small_m_5bit"         "$PY" bench/tests/test_small_m_5bit.py
  run "test_small_m_2bit"         "$PY" bench/tests/test_small_m_2bit.py
  run "test_dflash_ft"            "$PY" bench/tests/test_dflash_ft.py
  run "test_bonsai_loader"        "$PY" bench/tests/test_bonsai_loader.py
  run "test_dflash_prequantized --gpu" "$PY" bench/tests/test_dflash_prequantized.py --gpu
fi
if [ "${1:-}" = --models ]; then
  run "test_dflash_ft --with-models"    "$PY" bench/tests/test_dflash_ft.py --with-models
  run "test_bonsai_loader --with-models" "$PY" bench/tests/test_bonsai_loader.py --with-models
fi
exit $rc
