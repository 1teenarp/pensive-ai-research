#!/usr/bin/env python3
"""Bulk (or single) Hugging Face model downloader.

Patched from /trunk/ai/scripts/hf_bulk_download_v2.py (2026-09-14):
  - Token: no more copy/paste into ENV. Resolution order:
      1. $HF_TOKEN env var
      2. $HF_TOKEN_FILE (path to a file containing just the token)
      3. <script dir>/HF_TOKEN
      4. ~/.cache/huggingface/token
      5. interactive prompt (last resort)
    (On pensive, launch with HF_TOKEN_FILE=/trunk/ai/scripts/HF_TOKEN.)
  - Single-repo mode: pass a repo_id directly instead of a CSV.
      python hf_bulk_download_v3.py nvidia/GLM-5.3-Flash-NVFP4
      python hf_bulk_download_v3.py nvidia/GLM-5.3-Flash-NVFP4 --quant NVFP4
    (Omit --quant for whole-repo downloads; the NVFP4 repos ship quant
    in the repo name, not the filenames.)
  - CSV mode unchanged:
      python hf_bulk_download_v3.py models_v9.csv [--quant FALLBACK_QUANT]
"""
import os
import re
import sys
import getpass

# High-performance transfer: huggingface_hub >=1.x deprecated HF_HUB_ENABLE_HF_TRANSFER
# in favor of HF_XET_HIGH_PERFORMANCE (Xet). Keep the legacy var too, harmless if unused.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import snapshot_download

# --- CONSTANTS ---
BASE_DIR = "/trunk/ai/huggingface/models/"


def _token_candidates():
    cands = []
    env_file = os.getenv("HF_TOKEN_FILE")
    if env_file:
        cands.append(os.path.expanduser(env_file))
    cands.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "HF_TOKEN"))
    cands.append(os.path.expanduser("~/.cache/huggingface/token"))
    return cands


_HF_TOKEN_RE = re.compile(r"\bhf_[A-Za-z0-9]{16,}\b")


def _extract_token(text):
    """Return a bare hf_... token from text, or None.

    Handles both a bare token file and a Hugging Face "Export tokens" CSV
    (header `name, value` followed by rows like `hf_download, hf_abc...`),
    where the token value is the last hf_-prefixed field, not the name.
    """
    text = text.strip()
    if _HF_TOKEN_RE.fullmatch(text):
        return text
    # CSV export: take the last hf_-prefixed value on the last non-empty line
    matches = _HF_TOKEN_RE.findall(text)
    if matches:
        return matches[-1]
    return None


def get_hf_token():
    """Retrieve token from env, then a token file, else prompt safely."""
    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        extracted = _extract_token(token)
        if extracted:
            return extracted, "env HF_TOKEN"
    for path in _token_candidates():
        if os.path.isfile(path):
            with open(path) as f:
                extracted = _extract_token(f.read())
            if extracted:
                return extracted, path
            print(f"⚠️  {path} exists but no hf_... token found in it.")
    print("⚠️  HF token not found (env HF_TOKEN / HF_TOKEN_FILE / <script dir>/HF_TOKEN).")
    token = getpass.getpass("🔑 Enter your Hugging Face User Access Token: ").strip()
    if not token:
        print("❌ Error: A Hugging Face token is required. Exiting.")
        sys.exit(1)
    return token, "interactive"


def download_model(repo_id, quant=None, token=None):
    """Downloads a single model, mapping out clean structural layouts."""
    if "/" not in repo_id:
        provider = "misc"
        model_name = repo_id
    else:
        provider, model_name = repo_id.split("/", 1)

    target_dir = os.path.join(BASE_DIR, provider, model_name)
    quant = str(quant).strip() if quant else ""
    if quant and quant.lower() != "nan":
        target_dir = os.path.join(target_dir, quant)
        allow_patterns = [f"*{quant}*"]
        print(f"\n🚀 Target: {repo_id} (Quant: {quant}) -> {target_dir}")
    else:
        allow_patterns = None
        print(f"\n🚀 Target: {repo_id} (Full Model) -> {target_dir}")

    os.makedirs(target_dir, exist_ok=True)

    download_args = {
        "repo_id": repo_id,
        "local_dir": target_dir,
        "token": token,
        "max_workers": 8,
    }
    if allow_patterns:
        download_args["allow_patterns"] = allow_patterns + ["*.json", "*.md", "*.txt"]

    try:
        snapshot_download(**download_args)
        print(f"✅ Successfully downloaded {repo_id} to {target_dir}")
    except Exception as e:
        print(f"❌ Failed to download {repo_id}. Error: {e}")
        sys.exit(1)


def is_csv_path(p):
    return p.lower().endswith(".csv")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    quant = None
    if "--quant" in sys.argv:
        i = sys.argv.index("--quant")
        if i + 1 < len(sys.argv):
            quant = sys.argv[i + 1]
            args = [a for a in args if a != quant]

    if len(args) < 1:
        print("Usage: python hf_bulk_download_v3.py <repo_id|manifest.csv> [--quant QUANT]")
        sys.exit(1)

    token, src = get_hf_token()
    print(f"🔑 Using HF token from: {src}")

    target = args[0]
    if is_csv_path(target) and os.path.exists(target):
        import pandas as pd  # lazy: only needed for CSV mode
        df = pd.read_csv(target)
        print(f"📋 Found {len(df)} download targets in manifest.")
        for idx, row in df.iterrows():
            repo = row.get("repo_id")
            row_quant = row.get("quantization")
            if not repo or str(repo).strip() in ("", "nan"):
                continue
            if pd.isna(row_quant):
                row_quant = None
            row_quant = quant if quant is not None else row_quant
            download_model(str(repo).strip(), row_quant, token)
    else:
        download_model(target, quant, token)


def warn_no_hf_transfer():
    try:
        import hf_transfer  # noqa: F401
    except ImportError:
        print("⚠️  'hf_transfer' not installed. Download speeds will be limited.")
        print("   Install it with: pip install hf_transfer")
        print()


if __name__ == "__main__":
    warn_no_hf_transfer()
    main()
