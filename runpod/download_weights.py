"""Pre-download or verify Hugging Face weights required by TRELLIS.2-4B."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

MODEL_REPOS = [
    ("microsoft/TRELLIS.2-4B", "TRELLIS.2 4B checkpoints & pipeline.json", False),
    ("microsoft/TRELLIS-image-large", "Sparse structure decoder (ss_dec_conv3d_16l8_fp16)", False),
    ("facebook/dinov3-vitl16-pretrain-lvd1689m", "DINOv3 ViT-L/16 image feature extractor", True),
    ("briaai/RMBG-2.0", "BiRefNet background removal model", True),
]


def download_all(cache_dir: str | None = None, token: str | None = None) -> int:
    """Download all repositories referenced by TRELLIS.2-4B/pipeline.json."""
    resolved_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    resolved_cache = cache_dir or os.environ.get("HF_HOME")
    if resolved_cache:
        Path(resolved_cache).mkdir(parents=True, exist_ok=True)
        print(f"Using Hugging Face cache: {resolved_cache}")

    failed = False
    for repo_id, description, gated in MODEL_REPOS:
        print(f"\n==> Checking/downloading {repo_id} ({description})...")
        try:
            if repo_id == "microsoft/TRELLIS-image-large":
                # Only the sparse structure decoder checkpoint is referenced by TRELLIS.2-4B/pipeline.json
                for ext in ("json", "safetensors"):
                    path = hf_hub_download(
                        repo_id=repo_id,
                        filename=f"ckpts/ss_dec_conv3d_16l8_fp16.{ext}",
                        cache_dir=resolved_cache,
                        token=resolved_token,
                    )
                    print(f"    Cached: {path}")
            else:
                local_path = snapshot_download(
                    repo_id=repo_id,
                    cache_dir=resolved_cache,
                    token=resolved_token,
                )
                print(f"    Cached at: {local_path}")
        except GatedRepoError as exc:
            failed = True
            print(f"ERROR: Gated repository access denied for {repo_id}: {exc}", file=sys.stderr)
            print(
                f"  1. Visit https://huggingface.co/{repo_id} and request/accept access.\n"
                "  2. Export a valid read token as HF_TOKEN=hf_...",
                file=sys.stderr,
            )
        except HfHubHTTPError as exc:
            failed = True
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if gated and status in (401, 403):
                print(f"ERROR: Unauthorized ({status}) for gated repo {repo_id}. Check HF_TOKEN.", file=sys.stderr)
            else:
                print(f"ERROR: Failed to download {repo_id}: {exc}", file=sys.stderr)
        except Exception as exc:
            failed = True
            print(f"ERROR: Unexpected failure for {repo_id}: {exc}", file=sys.stderr)

    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=None, help="Override HF_HOME cache directory")
    parser.add_argument("--token", default=None, help="Hugging Face token (defaults to HF_TOKEN env var)")
    args = parser.parse_args()
    return download_all(cache_dir=args.cache_dir, token=args.token)


if __name__ == "__main__":
    sys.exit(main())
