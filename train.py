import os
import csv
import time
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    DATASET_DIR, IMAGE_HEIGHT, IMAGE_WIDTH, BATCH_SIZE,
    INSTANCES_PER_ID, EPOCHS, BASE_LEARNING_RATE, WARMUP_EPOCHS,
    DECAY_EPOCH_1, DECAY_EPOCH_2, DECAY_LR_1, DECAY_LR_2,
    WEIGHT_DECAY, LABEL_SMOOTHING, TRIPLET_MARGIN,
    CENTER_LOSS_WEIGHT, CENTER_LOSS_LEARNING_RATE,
    CHECKPOINT_DIR, CHECKPOINT_PATH, RESULTS_DIR, RESULTS_PATH,
    EVALUATE_EVERY, SEED, NUM_WORKERS, DEVICE,
)
from data import (
    build_transforms, Market1501TrainDataset,
    Market1501EvalDataset, RandomIdentityBatchSampler,
)
from model import ShuffleNetV2BNNeck
from losses import BatchHardTripletLoss, CenterLoss
from metrics import evaluate_reid, model_complexity


def get_device():
    if DEVICE == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    return torch.device(DEVICE)


def learning_rate_for_epoch(epoch):
    if epoch <= WARMUP_EPOCHS:
        return BASE_LEARNING_RATE * epoch / WARMUP_EPOCHS
    if epoch <= DECAY_EPOCH_1:
        return BASE_LEARNING_RATE
    if epoch <= DECAY_EPOCH_2:
        return DECAY_LR_1
    return DECAY_LR_2


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = get_device()
    print("Using device:", device)

    train_transform, eval_transform = build_transforms()
    train_dataset = Market1501TrainDataset(
        DATASET_DIR, train_transform
    )

    batch_sampler = RandomIdentityBatchSampler(
        train_dataset, BATCH_SIZE, INSTANCES_PER_ID
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    query_dataset = Market1501EvalDataset(
        os.path.join(DATASET_DIR, "query"), eval_transform
    )
    gallery_dataset = Market1501EvalDataset(
        os.path.join(DATASET_DIR, "bounding_box_test"), eval_transform
    )

    query_loader = DataLoader(
        query_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda")
    )
    gallery_loader = DataLoader(
        gallery_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda")
    )

    model = ShuffleNetV2BNNeck(
        num_classes=len(train_dataset.pid_to_label),
        use_last_stride_one=True,
    ).to(device)

    id_criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING
    )
    triplet_criterion = BatchHardTripletLoss(margin=TRIPLET_MARGIN)
    center_criterion = CenterLoss(
        len(train_dataset.pid_to_label), model.feature_dim
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    center_optimizer = torch.optim.SGD(
        center_criterion.parameters(),
        lr=CENTER_LOSS_LEARNING_RATE,
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    complexity = model_complexity(
        model,
        (1, 3, IMAGE_HEIGHT, IMAGE_WIDTH),
        device,
    )

    print("Model complexity for one image:")
    for key, value in complexity.items():
        print(f"{key}: {value:.4f}")

    start_epoch = 0
    if os.path.isfile(CHECKPOINT_PATH):
        checkpoint = torch.load(
            CHECKPOINT_PATH, map_location=device, weights_only=False
        )
        if checkpoint.get("pid_to_label") != train_dataset.pid_to_label:
            raise ValueError(
                "Checkpoint identity mapping differs from this dataset."
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        center_criterion.load_state_dict(checkpoint["center_loss_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        center_optimizer.load_state_dict(
            checkpoint["center_optimizer_state_dict"]
        )
        start_epoch = int(checkpoint["epoch"])
        print(f"Resuming strong-baseline checkpoint from epoch {start_epoch}.")
    else:
        print("No strong-baseline checkpoint found; training from scratch.")
        print(
            "The previous plain ShuffleNetV2 checkpoint is intentionally "
            "not overwritten because its architecture and training losses differ."
        )

    if not os.path.isfile(RESULTS_PATH):
        with open(
            RESULTS_PATH, "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.writer(file)
            writer.writerow([
                "Epoch", "Training Loss", "ID Loss", "Triplet Loss",
                "Center Loss", "Training Accuracy (%)",
                "Rank-1 (%)", "Rank-5 (%)", "Rank-10 (%)", "Rank-20 (%)",
                "mAP (%)", "Valid Queries", "Inference Parameters (M)",
                "Total Parameters (M)", "MACs (G)", "FLOPs (G)",
                "Learning Rate", "Training Time (seconds)",
            ])

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        lr = learning_rate_for_epoch(epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr

        model.train()
        center_criterion.train()

        totals = {
            "loss": 0.0, "id": 0.0, "triplet": 0.0, "center": 0.0
        }
        correct = 0
        total = 0
        start_time = time.time()

        for batch_index, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            center_optimizer.zero_grad(set_to_none=True)

            raw_features, bn_features, logits = model(images)

            id_loss = id_criterion(logits, labels)
            triplet_loss = triplet_criterion(raw_features, labels)
            center_loss = center_criterion(raw_features, labels)

            loss = (
                id_loss
                + triplet_loss
                + CENTER_LOSS_WEIGHT * center_loss
            )
            loss.backward()

            for parameter in center_criterion.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(1.0 / CENTER_LOSS_WEIGHT)

            optimizer.step()
            center_optimizer.step()

            batch_count = labels.size(0)
            totals["loss"] += loss.item() * batch_count
            totals["id"] += id_loss.item() * batch_count
            totals["triplet"] += triplet_loss.item() * batch_count
            totals["center"] += center_loss.item() * batch_count
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_count

            if batch_index % 200 == 0:
                print(
                    f"Epoch {epoch}/{EPOCHS} | "
                    f"Batch {batch_index}/{len(train_loader)}"
                )

        elapsed = time.time() - start_time
        train_loss = totals["loss"] / max(total, 1)
        id_loss_value = totals["id"] / max(total, 1)
        triplet_loss_value = totals["triplet"] / max(total, 1)
        center_loss_value = totals["center"] / max(total, 1)
        accuracy = 100.0 * correct / max(total, 1)

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Loss: {train_loss:.4f} "
            f"ID: {id_loss_value:.4f} "
            f"Triplet: {triplet_loss_value:.4f} "
            f"Center: {center_loss_value:.4f} "
            f"Training accuracy: {accuracy:.2f}% "
            f"LR: {lr:.7f} Time: {elapsed:.1f}s"
        )

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "center_loss_state_dict": center_criterion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "center_optimizer_state_dict": center_optimizer.state_dict(),
            "pid_to_label": train_dataset.pid_to_label,
            "feature_dim": model.feature_dim,
            "image_height": IMAGE_HEIGHT,
            "image_width": IMAGE_WIDTH,
            "complexity": complexity,
        }, CHECKPOINT_PATH)

        if epoch % EVALUATE_EVERY == 0:
            print(f"Evaluating retrieval metrics at epoch {epoch}...")
            metrics = evaluate_reid(
                model, query_loader, gallery_loader, device
            )

            print(
                f"Rank-1: {metrics['Rank-1']:.2f}% | "
                f"Rank-5: {metrics['Rank-5']:.2f}% | "
                f"Rank-10: {metrics['Rank-10']:.2f}% | "
                f"Rank-20: {metrics['Rank-20']:.2f}% | "
                f"mAP: {metrics['mAP']:.2f}%"
            )

            with open(
                RESULTS_PATH, "a", newline="", encoding="utf-8-sig"
            ) as file:
                writer = csv.writer(file)
                writer.writerow([
                    epoch,
                    round(train_loss, 5),
                    round(id_loss_value, 5),
                    round(triplet_loss_value, 5),
                    round(center_loss_value, 5),
                    round(accuracy, 2),
                    round(metrics["Rank-1"], 2),
                    round(metrics["Rank-5"], 2),
                    round(metrics["Rank-10"], 2),
                    round(metrics["Rank-20"], 2),
                    round(metrics["mAP"], 2),
                    metrics["Valid Queries"],
                    round(complexity["Inference Parameters (M)"], 4),
                    round(complexity["Total Parameters (M)"], 4),
                    round(complexity["MACs (G)"], 4),
                    round(complexity["FLOPs (G)"], 4),
                    lr,
                    round(elapsed, 1),
                ])

            print("CSV updated:", RESULTS_PATH)

    print("Training completed.")
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Evaluation CSV:", RESULTS_PATH)


if __name__ == "__main__":
    main()
