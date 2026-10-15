import os
import re
import random
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    INSTANCES_PER_ID,
    RANDOM_ERASING_PROBABILITY,
)


def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(10),
        transforms.RandomCrop((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        transforms.RandomErasing(
            p=RANDOM_ERASING_PROBABILITY,
            scale=(0.02, 0.4),
            ratio=(0.3, 3.3),
            value=0,
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, eval_transform


class Market1501TrainDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        folder = os.path.join(root, "bounding_box_train")

        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Training folder not found: {os.path.abspath(folder)}"
            )

        raw_samples = []
        for filename in sorted(os.listdir(folder)):
            match = re.match(r"(-?\d+)_c(\d+)", filename)
            if not match:
                continue

            pid = int(match.group(1))
            if pid <= 0:
                continue

            raw_samples.append((os.path.join(folder, filename), pid))

        if not raw_samples:
            raise RuntimeError("No valid Market-1501 training images found.")

        identities = sorted({pid for _, pid in raw_samples})
        self.pid_to_label = {
            pid: label for label, pid in enumerate(identities)
        }
        self.samples = [
            (path, self.pid_to_label[pid]) for path, pid in raw_samples
        ]

        self.indices_by_label = defaultdict(list)
        for index, (_, label) in enumerate(self.samples):
            self.indices_by_label[label].append(index)

        self.labels_with_samples = sorted(self.indices_by_label)

        print("Training images:", len(self.samples))
        print("Training identities:", len(identities))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)
        return image, label


class RandomIdentityBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, instances_per_id=4):
        if batch_size % instances_per_id != 0:
            raise ValueError(
                "BATCH_SIZE must be divisible by INSTANCES_PER_ID."
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.instances_per_id = instances_per_id
        self.identities_per_batch = batch_size // instances_per_id
        self.labels = list(dataset.labels_with_samples)
        self.indices_by_label = dataset.indices_by_label

        if len(self.labels) < self.identities_per_batch:
            raise ValueError("Not enough identities to construct a batch.")

    def __len__(self):
        return max(1, len(self.dataset) // self.batch_size)

    def __iter__(self):
        for _ in range(len(self)):
            selected_labels = random.sample(
                self.labels, self.identities_per_batch
            )
            batch_indices = []

            for label in selected_labels:
                indices = self.indices_by_label[label]

                if len(indices) >= self.instances_per_id:
                    chosen = random.sample(indices, self.instances_per_id)
                else:
                    chosen = random.choices(
                        indices, k=self.instances_per_id
                    )

                batch_indices.extend(chosen)

            yield batch_indices


class Market1501EvalDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.transform = transform
        self.samples = []

        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Evaluation folder not found: {os.path.abspath(folder)}"
            )

        for filename in sorted(os.listdir(folder)):
            match = re.match(r"(-?\d+)_c(\d+)", filename)
            if not match:
                continue

            pid = int(match.group(1))
            camid = int(match.group(2))
            self.samples.append(
                (os.path.join(folder, filename), pid, camid)
            )

        if not self.samples:
            raise RuntimeError(f"No valid images found in {folder}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, pid, camid = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)
        return image, pid, camid
