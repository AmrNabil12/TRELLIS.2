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

# Prefer persistent /workspace cache if mounted and writable; otherwise fall back to /root
if [[ -d /workspace ]] && [[ -w /workspace ]]; then
    export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
else
    export HF_HOME="/root/.cache/huggingface"
fi
mkdir -p "${HF_HOME}"

if [[ -z "${HF_TOKEN:-}" ]] && [[ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN is not set." >&2
    echo "  microsoft/TRELLIS.2-4B is public, but its pipeline.json loads two gated models:" >&2
    echo "    - facebook/dinov3-vitl16-pretrain-lvd1689m (requires manual HF access approval)" >&2
    echo "    - briaai/RMBG-2.0 (requires HF access acceptance)" >&2
    echo "  If weights are not already cached in ${HF_HOME}, startup loading will fail without HF_TOKEN." >&2
fi

exec uvicorn server:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
