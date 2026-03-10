import argparse
import json
import os
from typing import Optional

import torch

from main.torch_model import build_torch_model, normalize_model_name
from main.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a PyTorch checkpoint to ONNX for C++ inference."
    )
    parser.add_argument(
        "--model-name",
        default="ButterflyC",
        help="Model name without extension (default: ButterflyC).",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Path to .pt checkpoint. Defaults to manifest.torch_model_path or main/models/<model>.pt.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output ONNX path. Defaults to main/models/<model>.onnx.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (default: 18).",
    )
    parser.add_argument(
        "--add-softmax",
        action="store_true",
        help="Append softmax to outputs before export.",
    )
    return parser.parse_args()


def load_manifest(configs: dict) -> Optional[dict]:
    manifest_path = configs.get("manifest_path")
    if not manifest_path or not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_checkpoint(path: str):
    checkpoint = torch.load(path, map_location="cpu")
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def export_to_onnx(args):
    configs = load_config()
    manifest = load_manifest(configs)

    resolved_name = normalize_model_name(args.model_name)
    model_path = (
        args.checkpoint
        or (manifest.get("torch_model_path") if manifest else "")
        or os.path.join(configs["model_path"], f"{resolved_name}.pt")
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    image_size = (
        manifest.get("image_size")
        if manifest and "image_size" in manifest
        else int(configs.get("image_size", 224))
    )
    num_classes = (
        manifest.get("num_classes")
        if manifest and "num_classes" in manifest
        else int(configs.get("num_classes", 75))
    )

    _, model, _ = build_torch_model(
        resolved_name,
        num_classes=num_classes,
        pretrained=False,
    )
    state_dict = load_checkpoint(model_path)
    model.load_state_dict(state_dict)
    model.eval()

    if args.add_softmax:
        model = torch.nn.Sequential(model, torch.nn.Softmax(dim=1))

    dummy = torch.randn(1, 3, image_size, image_size)
    output_path = (
        args.output
        or os.path.join(configs["model_path"], f"{resolved_name}.onnx")
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    opset = int(args.opset or 18)
    if opset < 18:
        print(
            "Requested opset < 18. Upgrading to opset 18 to avoid exporter version conversion."
        )
        opset = 18

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["images"],
        output_names=["probs" if args.add_softmax else "logits"],
        dynamic_axes={
            "images": {0: "batch"},
            "probs" if args.add_softmax else "logits": {0: "batch"},
        },
        opset_version=opset,
    )
    print(f"Exported ONNX to {output_path}")


def main():
    args = parse_args()
    export_to_onnx(args)


if __name__ == "__main__":
    main()
