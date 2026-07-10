"""Upload fitted lenses + provenance to a HF Hub model repo.

Usage (requires `huggingface-cli login` once, owner's account):
    python scripts/upload_lenses.py --repo <hf-username>/jlens-scaling-lenses
"""

from __future__ import annotations

import argparse
import glob
import os

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    for slug_dir in sorted(glob.glob(os.path.join(args.artifacts, "*"))):
        if not os.path.isdir(slug_dir):
            continue
        slug = os.path.basename(slug_dir)
        for name in ("lens.pt", "fit_meta.json"):
            path = os.path.join(slug_dir, name)
            if os.path.exists(path):
                api.upload_file(
                    path_or_fileobj=path,
                    path_in_repo=f"{slug}/{name}",
                    repo_id=args.repo,
                )
                print(f"uploaded {slug}/{name}")


if __name__ == "__main__":
    main()
