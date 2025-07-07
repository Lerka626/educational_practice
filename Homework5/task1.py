import os
import random
import shutil
from pathlib import Path

import torch
from torchvision import transforms
from datasets import CustomImageDataset
from utils import (
    show_single_augmentation,
    show_multiple_augmentations,
)
from PIL import Image
import matplotlib.pyplot as plt


data_dir = Path("C:/Users/Lerik/OneDrive/Desktop/all_practices/Practice5/data")
train_dir = data_dir / "train"
val_dir = data_dir / "val"
result_dir = Path("plots")
result_dir.mkdir(exist_ok=True)

val_split = 0.2  


def split_train_val(train_dir: Path, val_dir: Path, val_split: float = 0.2):
    """
    Делает разбиение датасета на train и val.
    Если val_dir уже существует, пропускает.
    """
    if val_dir.exists():
        print("Папка val/ уже существует, пропускаю разделение.")
        return

    for class_dir in train_dir.iterdir():
        if not class_dir.is_dir():
            continue

        val_class_dir = val_dir / class_dir.name
        val_class_dir.mkdir(parents=True, exist_ok=True)

        images = list(class_dir.glob("*"))
        random.shuffle(images)
        val_count = int(len(images) * val_split)

        val_images = images[:val_count]

        for img_path in val_images:
            shutil.move(str(img_path), str(val_class_dir / img_path.name))

    print("Разделение train/val завершено.")


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


def define_augmentations():
    """
    Возвращает список отдельных аугментаций и комбинированную.
    """
    torchvision_augs = [
        ("RandomHorizontalFlip", transforms.RandomHorizontalFlip(p=1.0)),
        ("RandomCrop", transforms.RandomCrop(200, padding=20)),
        ("ColorJitter", transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)),
        ("RandomRotation", transforms.RandomRotation(degrees=30)),
        ("RandomGrayscale", transforms.RandomGrayscale(p=1.0)),
    ]

    combined_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(200, padding=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.RandomGrayscale(p=0.2),
    ])

    return torchvision_augs, combined_aug


def apply_and_save_augmentations(img: torch.Tensor, label: str, torchvision_augs, combined_aug):
    """
    Применяет аугментации к одному изображению и сохраняет результаты в папку results/.
    Также показывает результаты на экране.
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    for aug_name, aug in torchvision_augs:
        pil_img = to_pil(img)
        aug_img = to_tensor(aug(pil_img))

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(to_pil(img))
        axes[0].set_title("Оригинал")
        axes[0].axis('off')
        axes[1].imshow(to_pil(aug_img))
        axes[1].set_title(f"{label} - {aug_name}")
        axes[1].axis('off')

        plt.tight_layout()
        plt_path = result_dir / f"{label}_aug_{aug_name}.png"
        fig.savefig(plt_path)
        print(f"Сохранено: {plt_path}")
        plt.show()
        plt.close(fig)

    # Все вместе
    aug_imgs = []
    aug_titles = []
    for aug_name, aug in torchvision_augs:
        pil_img = to_pil(img)
        aug_imgs.append(to_tensor(aug(pil_img)))
        aug_titles.append(aug_name)

    n = len(aug_imgs) + 1
    fig, axes = plt.subplots(1, n, figsize=(2*n, 2))
    axes[0].imshow(to_pil(img))
    axes[0].set_title("Оригинал")
    axes[0].axis('off')

    for i, (aug_img, title) in enumerate(zip(aug_imgs, aug_titles), 1):
        axes[i].imshow(to_pil(aug_img))
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.tight_layout()
    plt_path = result_dir / f"{label}_all_augs.png"
    fig.savefig(plt_path)
    print(f"Сохранено: {plt_path}")
    plt.show()
    plt.close(fig)

    # Комбинированные
    pil_img = to_pil(img)
    combined_imgs = [to_tensor(combined_aug(pil_img)) for _ in range(4)]

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    axes[0].imshow(pil_img)
    axes[0].set_title("Оригинал")
    axes[0].axis('off')

    for i, combined_img in enumerate(combined_imgs, 1):
        axes[i].imshow(to_pil(combined_img))
        axes[i].set_title(f"Combined #{i}")
        axes[i].axis('off')

    plt.tight_layout()
    plt_path = result_dir / f"{label}_combined.png"
    fig.savefig(plt_path)
    print(f"Сохранено: {plt_path}")
    plt.show() 
    plt.close(fig)


def main():
    """
    Основная функция для выполнения задания 1:
    - train/val split
    - выбор 5 изображений
    - применение и сохранение аугментаций
    """
    split_train_val(train_dir, val_dir, val_split=val_split)

    dataset = CustomImageDataset(
        root_dir=str(train_dir),
        transform=transforms.ToTensor(),
        target_size=(224, 224),
    )

    selected_images = select_images_from_different_classes(dataset, num_classes=5)

    torchvision_augs, combined_aug = define_augmentations()

    for img, label in selected_images:
        print(f"Класс: {label}")
        apply_and_save_augmentations(img, label, torchvision_augs, combined_aug)


if __name__ == "__main__":
    main()


