import argparse
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import DryRunError, HfHubHTTPError, LocalEntryNotFoundError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the K2 base model from Hugging Face.")
    parser.add_argument("--model-id", default="daven3/k2", help="Hugging Face model id to download.")
    parser.add_argument(
        "--local-dir",
        default=str(Path(__file__).parent / "models" / "k2"),
        help="Directory where model files will be stored.",
    )
    parser.add_argument("--revision", default=None, help="Optional model revision, branch, or commit.")
    parser.add_argument("--token", default=None, help="Optional Hugging Face token.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        path = snapshot_download(
            repo_id=args.model_id,
            revision=args.revision,
            local_dir=local_dir,
            token=args.token,
            dry_run=args.dry_run,
            max_workers=8,
        )
    except (DryRunError, LocalEntryNotFoundError) as exc:
        raise SystemExit(
            "Could not reach Hugging Face. Check your internet connection, proxy/DNS, "
            "or try the same command on a machine with network access."
        ) from exc
    except HfHubHTTPError as exc:
        raise SystemExit(
            "Hugging Face rejected the request. If this is a gated/private model, run "
            "`hf auth login` or pass `--token <HF_TOKEN>`."
        ) from exc

    print(path)


if __name__ == "__main__":
    main()
