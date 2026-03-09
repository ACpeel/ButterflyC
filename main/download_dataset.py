import argparse
import shutil
from pathlib import Path


REQUIRED_ENTRIES = (
    "Training_set.csv",
    "Testing_set.csv",
    "train",
    "test",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and prepare the butterfly dataset via kagglehub."
    )
    parser.add_argument(
        "--dataset",
        default="phucthaiv02/butterfly-image-classification",
        help="Kaggle dataset slug, default: phucthaiv02/butterfly-image-classification",
    )
    parser.add_argument(
        "--output",
        default="data",
        help="Project-relative output directory, default: data",
    )
    return parser.parse_args()


def find_dataset_root(download_path):
    root = Path(download_path).resolve()
    if has_required_entries(root):
        return root

    for candidate in root.rglob("*"):
        if candidate.is_dir() and has_required_entries(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not find Training_set.csv, Testing_set.csv, train/, and test/ "
        f"under downloaded path: {root}"
    )


def has_required_entries(directory):
    return all((directory / entry).exists() for entry in REQUIRED_ENTRIES)


def sync_dataset(dataset_root, output_root):
    output_root.mkdir(parents=True, exist_ok=True)

    for entry in REQUIRED_ENTRIES:
        source = dataset_root / entry
        destination = output_root / entry

        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)


def main():
    try:
        import kagglehub
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "kagglehub is not installed. Run `uv sync` first."
        ) from error

    args = parse_args()
    output_root = Path(args.output).resolve()
    download_path = Path(kagglehub.dataset_download(args.dataset)).resolve()
    dataset_root = find_dataset_root(download_path)
    sync_dataset(dataset_root, output_root)

    print(f"Downloaded dataset cache: {download_path}")
    print(f"Prepared project dataset: {output_root}")
    print("Expected files:")
    for entry in REQUIRED_ENTRIES:
        print(f"  - {output_root / entry}")


if __name__ == "__main__":
    main()
