import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

train_dir = Path("C:/Users/Lerik/OneDrive/Desktop/all_practices/Practice5/data/train")
result_dir = Path("C:/Users/Lerik/OneDrive/Desktop/all_practices/Practice5/plots")
result_dir.mkdir(exist_ok=True)


def analyze_dataset(train_dir: Path, result_dir: Path):
    """
    Анализирует датасет:
    - подсчитывает количество изображений в каждом классе
    - вычисляет мин/макс/средние размеры изображений
    - визуализирует распределения
    Все результаты сохраняет в result_dir.
    """
    class_image_counts = {}
    all_widths = []
    all_heights = []

    # Перебираю классы (папки)
    for class_folder in train_dir.iterdir():
        if not class_folder.is_dir():
            continue

        images = list(class_folder.glob("*"))
        class_image_counts[class_folder.name] = len(images)

        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    all_widths.append(w)
                    all_heights.append(h)
            except Exception as e:
                print(f"Ошибка при открытии {img_path}: {e}")

    # Подсчет статистики размеров
    widths = np.array(all_widths)
    heights = np.array(all_heights)

    print("Количество изображений по классам:")
    for cls, count in class_image_counts.items():
        print(f"  {cls}: {count}")

    print("\nРазмеры изображений (ширина x высота):")
    print(f"  Минимальный размер: {widths.min()} x {heights.min()}")
    print(f"  Максимальный размер: {widths.max()} x {heights.max()}")
    print(f"  Средний размер: {widths.mean():.1f} x {heights.mean():.1f}")

    # Визуализация 1: гистограмма по классам
    plt.figure(figsize=(10, 6))
    classes = list(class_image_counts.keys())
    counts = list(class_image_counts.values())
    plt.bar(classes, counts, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.title("Количество изображений по классам")
    plt.ylabel("Количество")
    plt.tight_layout()
    plt.savefig(result_dir / "image_counts_per_class.png")
    plt.close()

    # Визуализация 2: гистограммы размеров
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=30, color='salmon', alpha=0.7)
    plt.title("Распределение ширины изображений")
    plt.xlabel("Ширина (пиксели)")
    plt.ylabel("Количество")

    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=30, color='lightgreen', alpha=0.7)
    plt.title("Распределение высоты изображений")
    plt.xlabel("Высота (пиксели)")
    plt.ylabel("Количество")

    plt.tight_layout()
    plt.savefig(result_dir / "image_size_distribution.png")
    plt.close()

    # Визуализация 3: scatter plot размеров (ширина vs высота)
    plt.figure(figsize=(8, 6))
    plt.scatter(widths, heights, alpha=0.3, s=10, c='purple')
    plt.title("Распределение размеров изображений (ширина vs высота)")
    plt.xlabel("Ширина (пиксели)")
    plt.ylabel("Высота (пиксели)")
    plt.grid(True)
    plt.savefig(result_dir / "image_size_scatter.png")
    plt.close()

if __name__ == "__main__":
    analyze_dataset(train_dir, result_dir)

