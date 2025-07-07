import time
import tracemalloc
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms

train_dir = Path("C:/Users/Lerik/OneDrive/Desktop/all_practices/Practice5/data/train")
result_dir = Path("C:/Users/Lerik/OneDrive/Desktop/all_practices/Practice5/plots")
result_dir.mkdir(exist_ok=True)

N_IMAGES = 100
SIZES = [64, 128, 224, 512]


def collect_image_paths(root_dir: Path, n: int) -> list[Path]:
    """
    Собирает пути до первых `n` изображений из папки train/.
    """
    paths = []
    for class_dir in root_dir.iterdir():
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.glob("*"):
            paths.append(img_path)
            if len(paths) >= n:
                break
        if len(paths) >= n:
            break
    print(f"Найдено {len(paths)} изображений для эксперимента.")
    return paths


def measure_time_and_memory(image_paths: list[Path], size: int) -> tuple[float, float]:
    """
    Применяет ресайз + аугментации к списку изображений заданного размера.
    Возвращает кортеж (время_сек, пиковая_память_МБ)
    """
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    start_time = time.time()
    tracemalloc.start()

    for img_path in image_paths:
        with Image.open(img_path) as img:
            transformed = transform(img)

    current, peak = tracemalloc.get_traced_memory()
    elapsed = time.time() - start_time
    tracemalloc.stop()

    mem_mb = peak / (1024 * 1024)
    return elapsed, mem_mb


def plot_metric_vs_size(sizes, values, ylabel, title, filename, color="blue"):
    """
    Строит график зависимости метрики от размера.
    """
    plt.figure()
    plt.plot(sizes, values, marker='o', color=color)
    plt.title(title)
    plt.xlabel("Размер изображения (px)")
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(result_dir / filename)
    plt.close()
    print(f"Сохранено: {filename}")


def run_experiment():
    """
    Запускает эксперимент:
    - по всем размерам
    - измеряет время и память
    - строит графики
    """
    image_paths = collect_image_paths(train_dir, N_IMAGES)

    times = []
    memories = []

    for size in SIZES:
        print(f"Эксперимент для размера {size}x{size}")
        elapsed, mem_mb = measure_time_and_memory(image_paths, size)
        times.append(elapsed)
        memories.append(mem_mb)
        print(f"Время: {elapsed:.2f} сек | Память: {mem_mb:.2f} МБ")

    plot_metric_vs_size(
        SIZES, times,
        ylabel="Время (сек)",
        title="Время загрузки и аугментаций от размера",
        filename="experiment_time_vs_size.png"
    )

    plot_metric_vs_size(
        SIZES, memories,
        ylabel="Память (МБ)",
        title="Потребление памяти от размера",
        filename="experiment_memory_vs_size.png",
        color="red"
    )

    print("Эксперимент завершен. Графики сохранены в папку results/")


if __name__ == "__main__":
    run_experiment()

