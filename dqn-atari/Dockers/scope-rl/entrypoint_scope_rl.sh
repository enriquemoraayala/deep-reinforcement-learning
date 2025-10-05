#!/usr/bin/env bash
# script para poner la libreria scope_rl en modo editable y que pueda hacer debug

set -e

# Si el usuario montó su clon en /opt/scope-rl, instala editable automáticamente
if [ -d "${SCOPE_RL_DEV_PATH}" ] && [ "$(ls -A ${SCOPE_RL_DEV_PATH})" ]; then
  echo "[entrypoint_scope_rl] Detected scope-rl source at ${SCOPE_RL_DEV_PATH} -> installing editable..."
  # --no-build-isolation evita sorpresas con pyproject
  pip install -e "${SCOPE_RL_DEV_PATH}" --no-build-isolation || {
    echo "[entrypoint_scope_rl] WARNING: editable install failed; continuing with PyPI installation"
  }
else
  echo "[entrypoint_scope_rl] No dev source at ${SCOPE_RL_DEV_PATH}. Using PyPI-installed scope-rl."
fi

exec "$@"
