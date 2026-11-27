#!/usr/bin/env bash
set -euo pipefail

# Enable RunPod SSH access when PUBLIC_KEY is injected
if [[ -n "${PUBLIC_KEY:-}" ]] && [[ -x /usr/sbin/sshd ]]; then
    mkdir -p /root/.ssh /run/sshd
    chmod 700 /root/.ssh
    printf '%s\n' "${PUBLIC_KEY}" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    /usr/sbin/sshd || echo "Warning: failed to start sshd" >&2
fi

# Detect persistent storage: /runpod-volume (Serverless) > /workspace (Pods)
STORAGE_ROOT=""
if [[ -d /runpod-volume ]] && [[ -w /runpod-volume ]]; then
    STORAGE_ROOT="/runpod-volume"
elif [[ -d /workspace ]] && [[ -w /workspace ]]; then
    STORAGE_ROOT="/workspace"
fi

if [[ -n "${STORAGE_ROOT}" ]]; then
    export HF_HOME="${HF_HOME:-${STORAGE_ROOT}/hf_cache}"
    export TRELLIS2_JOBS_DIR="${TRELLIS2_JOBS_DIR:-${STORAGE_ROOT}/jobs}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${STORAGE_ROOT}/triton_cache}"
    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${STORAGE_ROOT}/torchinductor_cache}"
    export FLEX_GEMM_AUTOTUNE_CACHE_PATH="${FLEX_GEMM_AUTOTUNE_CACHE_PATH:-${STORAGE_ROOT}/flex_gemm_autotune_cache.json}"
    mkdir -p "${TRELLIS2_JOBS_DIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
    echo "[start.sh] STORAGE_ROOT=${STORAGE_ROOT} (persistent volume mounted)"
else
    export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/.cache/triton}"
    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/root/.cache/torchinductor}"
    export FLEX_GEMM_AUTOTUNE_CACHE_PATH="${FLEX_GEMM_AUTOTUNE_CACHE_PATH:-/root/.cache/flex_gemm_autotune_cache.json}"
    mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
    echo "[start.sh] STORAGE_ROOT=none (/runpod-volume and /workspace unavailable; using container disk)"
fi
export TORCHINDUCTOR_FX_GRAPH_CACHE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}"
export TORCHINDUCTOR_AUTOGRAD_CACHE="${TORCHINDUCTOR_AUTOGRAD_CACHE:-1}"
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE="${FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE:-1}"
mkdir -p "${HF_HOME}"
echo "[start.sh] HF_HOME=${HF_HOME} TRITON_CACHE_DIR=${TRITON_CACHE_DIR} TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR} TRELLIS2_JOBS_DIR=${TRELLIS2_JOBS_DIR:-<default>}"
df -h / "${STORAGE_ROOT:-/}" 2>/dev/null || true
if [[ -n "${STORAGE_ROOT}" ]]; then
    ls -la "${STORAGE_ROOT}" 2>/dev/null || true
fi

if [[ -z "${HF_TOKEN:-}" ]] && [[ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN is not set." >&2
    echo "  microsoft/TRELLIS.2-4B is public, but its pipeline.json loads two gated models:" >&2
    echo "    - facebook/dinov3-vitl16-pretrain-lvd1689m (requires manual HF access approval)" >&2
    echo "    - briaai/RMBG-2.0 (requires HF access acceptance)" >&2
    echo "  If weights are not already cached in ${HF_HOME}, startup loading will fail without HF_TOKEN." >&2
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"

exec uvicorn server:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
