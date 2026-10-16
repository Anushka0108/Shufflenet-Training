
import os
import torch
from torch.utils.data import DataLoader

from config import (
    DATASET_DIR,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    NUM_WORKERS,
    DEVICE,
)
from data import build_transforms, Market1501EvalDataset
from model import ShuffleNetV2BNNeck
from metrics import evaluate_reid, model_complexity


def main():
    if DEVICE == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(DEVICE)

    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    train_transform, eval_transform = build_transforms()

    query_dataset = Market1501EvalDataset(
        os.path.join(DATASET_DIR, "query"), eval_transform
    )
    gallery_dataset = Market1501EvalDataset(
        os.path.join(DATASET_DIR, "bounding_box_test"),
        eval_transform,
    )

    query_loader = DataLoader(
        query_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = ShuffleNetV2BNNeck(
        num_classes=len(checkpoint["pid_to_label"]),
        use_last_stride_one=True,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Evaluating checkpoint from epoch {checkpoint['epoch']}")

    complexity = model_complexity(
        model, (1, 3, IMAGE_HEIGHT, IMAGE_WIDTH), device
    )

    metrics = evaluate_reid(
        model, query_loader, gallery_loader, device
    )

    print("\nModel complexity:")
    for key, value in complexity.items():
        print(f"{key}: {value:.4f}")

    print("\nFinal retrieval results:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
