import os
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

train_dir = Path("C:/Users/Lerik/OneDrive/Desktop/all_practices/Practice5/data/train")
result_dir = Path("C:/Users/Lerik/OneDrive/Desktop/all_practices/Practice5/results")
result_dir.mkdir(exist_ok=True)

class AugmentationPipeline:
    """
    Класс для пайплайна аугментаций.
    Позволяет добавлять, убирать, применять аугментации к изображениям.
    """
    def __init__(self):
        # список кортежей (имя, трансформация)
        self.augmentations = []

    def add_augmentation(self, name, aug):
        """
        Добавить аугментацию.
        :param name: имя аугментации
        :param aug: объект torchvision.transforms
        """
        self.augmentations.append((name, aug))

    def remove_augmentation(self, name):
        """
        Убрать аугментацию по имени.
        """
        self.augmentations = [(n, a) for n, a in self.augmentations if n != name]

    def apply(self, image):
        """
        Применить все аугментации по очереди к изображению PIL.
        :param image: PIL.Image
        :return: PIL.Image после всех трансформаций
        """
        pil_image = image.copy()
        for _, aug in self.augmentations:
            pil_image = aug(pil_image)
        return pil_image

    def get_augmentations(self):
        """
        Получить список имен аугментаций.
        """
        return [name for name, _ in self.augmentations]


def create_pipeline(config="light"):
    """
    Создать пайплайн с предустановленной конфигурацией.
    """
    pipeline = AugmentationPipeline()

    if config == "light":
        pipeline.add_augmentation("RandomHorizontalFlip", transforms.RandomHorizontalFlip(p=0.5))

    elif config == "medium":
        pipeline.add_augmentation("RandomHorizontalFlip", transforms.RandomHorizontalFlip(p=0.5))
        pipeline.add_augmentation("RandomCrop", transforms.RandomCrop(200, padding=20))
        pipeline.add_augmentation("ColorJitter", transforms.ColorJitter(brightness=0.2, contrast=0.2))

    elif config == "heavy":
        pipeline.add_augmentation("RandomHorizontalFlip", transforms.RandomHorizontalFlip(p=0.5))
        pipeline.add_augmentation("RandomCrop", transforms.RandomCrop(180, padding=30))
        pipeline.add_augmentation("ColorJitter", transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4))
        pipeline.add_augmentation("RandomRotation", transforms.RandomRotation(30))
        pipeline.add_augmentation("RandomGrayscale", transforms.RandomGrayscale(p=0.3))

    else:
        raise ValueError(f"Неизвестная конфигурация: {config}")

    return pipeline


def apply_pipeline_to_train(train_dir, result_dir, pipeline: AugmentationPipeline, config_name: str):
    """
    Применяет пайплайн к train и сохраняет результаты в results/augmented_{config_name}/
    """
    output_dir = result_dir / f"augmented_{config_name}"
    output_dir.mkdir(exist_ok=True)

    for class_folder in train_dir.iterdir():
        if not class_folder.is_dir():
            continue

        class_output_dir = output_dir / class_folder.name
        class_output_dir.mkdir(parents=True, exist_ok=True)

        for img_path in class_folder.glob("*"):
            try:
                with Image.open(img_path) as img:
                    aug_img = pipeline.apply(img)
                    save_path = class_output_dir / img_path.name
                    aug_img.save(save_path)
            except Exception as e:
                print(f"Ошибка при обработке {img_path}: {e}")

    print(f"Результаты конфигурации '{config_name}' сохранены в {output_dir}")


if __name__ == "__main__":
    configs = ["light", "medium", "heavy"]

    for config_name in configs:
        print(f"Применение конфигурации: {config_name}")
        pipeline = create_pipeline(config_name)
        print("Аугментации в пайплайне:", pipeline.get_augmentations())
        apply_pipeline_to_train(train_dir, result_dir, pipeline, config_name)

    print("Все конфигурации успешно применены и сохранены в папку results/")

