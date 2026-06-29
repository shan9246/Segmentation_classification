import os
import cv2
import numpy as np

from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms


CLASSES = {
    "benign": 0,
    "malignant": 1,
    "normal": 2
}


class BUSIDataset(Dataset):

    def __init__(self, samples, image_size=224):

        self.samples = samples
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        #################################################
        # Read image
        #################################################

        image = cv2.imread(sample["image"])
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = cv2.resize(
            image,
            (self.image_size, self.image_size)
        )

        #################################################
        # Read mask
        #################################################

        if sample["mask"] is not None:

            mask = cv2.imread(
                sample["mask"],
                cv2.IMREAD_GRAYSCALE
            )

        else:

            mask = np.zeros(
                (image.shape[0], image.shape[1]),
                dtype=np.uint8
            )

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        #################################################
        # Binary mask
        #################################################

        mask = (mask > 0).astype(np.float32)

        #################################################
        # Convert to tensor
        #################################################

        image = self.transform(image)

        mask = torch.from_numpy(mask).float()

        mask = mask.unsqueeze(0)

        label = torch.tensor(
            sample["label"],
            dtype=torch.long
        )

        return {
            "image": image,
            "mask": mask,
            "label": label,
            "class_name": sample["class_name"]
        }
def prepare_busi_dataset(
        dataset_root,
        test_size=0.15,
        val_size=0.15,
        random_state=42):

    samples = []

    for class_name, label in CLASSES.items():

        class_folder = os.path.join(
            dataset_root,
            class_name
        )

        for file_name in os.listdir(class_folder):

            if "_mask" in file_name:
                continue

            image_path = os.path.join(
                class_folder,
                file_name
            )

            name, ext = os.path.splitext(file_name)

            mask_path = os.path.join(
                class_folder,
                f"{name}_mask{ext}"
            )

            if not os.path.exists(mask_path):
                mask_path = None

            samples.append({
                "image": image_path,
                "mask": mask_path,
                "label": label,
                "class_name": class_name
            })

    labels = [s["label"] for s in samples]

    train_samples, temp_samples = train_test_split(
        samples,
        test_size=(test_size + val_size),
        stratify=labels,
        random_state=random_state
    )

    temp_labels = [s["label"] for s in temp_samples]

    val_ratio = val_size / (test_size + val_size)

    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=(1 - val_ratio),
        stratify=temp_labels,
        random_state=random_state
    )

    return (
        train_samples,
        val_samples,
        test_samples
    )


def get_distribution(samples):

    counter = Counter(
        [s["class_name"] for s in samples]
    )

    return counter