import random
from pathlib import Path

import torch
from torchvision import transforms
from datasets import CustomImageDataset
from utils import show_single_augmentation, show_multiple_augmentations

from torchvision import transforms

def get_extra_augs():
    """
    Возвращает готовые аугментации для сравнения.
    """
    extra_augs = [
        ("HorizontalFlip", transforms.RandomHorizontalFlip(p=1.0)),
        ("VerticalFlip", transforms.RandomVerticalFlip(p=1.0)),
        ("RandomResizedCrop", transforms.RandomResizedCrop(224)),
    ]
    return extra_augs

from custom_augs_ForTask2 import (
    RandomBlur,
    RandomPerspectiveCustom,
    RandomBrightnessContrast,
)

from PIL import Image

import matplotlib.pyplot as plt


data_dir = Path("C:/Users/Lerik/OneDrive/Desktop/all_practices/Practice5/data")
train_dir = data_dir / "train"
result_dir = Path("C:/Users/Lerik/OneDrive/Desktop/all_practices/Practice5/results_task2")
result_dir.mkdir(exist_ok=True)


def select_images_from_different_classes(dataset, num_classes: int = 5):
    """
    Выбирает по одному изображению из разных классов.
    Возвращает список (img, label_str)
    """
    class_names = dataset.get_class_names()
    selected = []
    seen_classes = set()
    i = 0

    while len(seen_classes) < num_classes and i < len(dataset):
        img, label = dataset[i]
        if label not in seen_classes:
            seen_classes.add(label)
            selected.append((img, class_names[label]))
        i += 1

    print(f"Выбраны классы: {[label for _, label in selected]}")
    return selected


def define_custom_augmentations():
    """
    Определяет 3 кастомные аугментации.
    """
    custom_augs = [
        ("RandomBlur", RandomBlur(p=1.0)),
        ("RandomPerspective", RandomPerspectiveCustom(distortion_scale=0.5, p=1.0)),
        ("RandomBrightnessContrast", RandomBrightnessContrast(p=1.0)),
    ]
    return custom_augs


def apply_and_compare_augmentations(img: torch.Tensor, label: str, custom_augs, extra_augs):
    """
    Применяет кастомные и готовые аугментации, сравнивает и сохраняет результаты.
    """
    pil_img = transforms.ToPILImage()(img)

    for aug_name, aug in custom_augs:
        custom_img = transforms.ToTensor()(aug(pil_img))
        extra_aug = dict(extra_augs).get(aug_name)

        if extra_aug:
            extra_img = transforms.ToTensor()(extra_aug(pil_img))

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(img.permute(1, 2, 0))
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(custom_img.permute(1, 2, 0))
            axes[1].set_title(f"Custom: {aug_name}")
            axes[1].axis("off")

            axes[2].imshow(extra_img.permute(1, 2, 0))
            axes[2].set_title(f"Extra: {aug_name}")
            axes[2].axis("off")

            plt.tight_layout()

            plt_path = result_dir / f"{label}_compare_{aug_name}.png"
            plt.savefig(plt_path)
            print(f"Сохранено: {plt_path}")
            plt.close(fig)

        else:
            print(f"Для {aug_name} нет аналога в extra_augs")


def main():
    """
    Задание 2:
    - Выбор 5 изображений
    - Применение кастомных аугментаций
    - Сравнение с готовыми
    """
    dataset = CustomImageDataset(
        root_dir=str(train_dir),
        transform=transforms.ToTensor(),
        target_size=(224, 224),
    )

    selected_images = select_images_from_different_classes(dataset, num_classes=5)

    custom_augs = define_custom_augmentations()
    extra_augs = get_extra_augs()

    for img, label in selected_images:
        print(f"Класс: {label}")
        apply_and_compare_augmentations(img, label, custom_augs, extra_augs)


if __name__ == "__main__":
    main()
