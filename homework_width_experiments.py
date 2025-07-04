# # 2.1 Сравнение моделей разной ширины

# import torch
# import time
# import logging
# from datasets import get_mnist_loaders
# from models import FullyConnectedModel
# from trainer import train_model
# from utils import plot_training_history, count_parameters

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Настройка логирования:
# logging.basicConfig(
#     level=logging.INFO,
#     format='[%(asctime)s] %(message)s',
#     handlers=[
#         logging.FileHandler("width_experiment.log"),
#         logging.StreamHandler()
#     ]
# )

# def build_layers(widths):
#     """
#     Создает конфигурацию слоев для заданной ширины.

#     :param widths: список ширин скрытых слоев
#     :return: список словарей конфигурации
#     """
#     layers = []
#     for width in widths:
#         layers.append({"type": "linear", "size": width})
#         layers.append({"type": "relu"})
#     return layers


# def train_and_evaluate(width_name, widths, train_loader, test_loader, input_size, num_classes, epochs):
#     """
#     Обучает и оценивает модель с заданной шириной слоев.

#     :param width_name: название варианта ширины
#     :param widths: список ширин скрытых слоев
#     """
#     logging.info(f"Начало обучения: {width_name} — ширины {widths}")

#     layers = build_layers(widths)
#     model = FullyConnectedModel(input_size=input_size, num_classes=num_classes, layers=layers).to(device)

#     n_params = count_parameters(model)
#     logging.info(f"Параметров в модели ({width_name}): {n_params}")

#     start_time = time.time()
#     history = train_model(model, train_loader, test_loader, epochs=epochs, device=device)
#     elapsed_time = time.time() - start_time

#     logging.info(f"Время обучения ({width_name}): {elapsed_time:.2f} сек")
#     logging.info(f"Тестовая точность ({width_name}): {history['test_accs'][-1]:.4f}")
#     logging.info("-" * 40)

#     plot_training_history(history)
#     return {
#         "width_name": width_name,
#         "widths": widths,
#         "n_params": n_params,
#         "final_test_acc": history['test_accs'][-1],
#         "time": elapsed_time
#     }


# def run_width_experiments():
#     """
#     Запускает эксперимент по сравнению моделей разной ширины.
#     """
#     batch_size = 64
#     epochs = 10
#     input_size = 28 * 28
#     num_classes = 10

#     train_loader, test_loader = get_mnist_loaders(batch_size)

#     configs = {
#         "Узкие": [64, 32, 16],
#         "Средние": [256, 128, 64],
#         "Широкие": [1024, 512, 256],
#         "Очень широкие": [2048, 1024, 512]
#     }

#     results = []

#     for name, widths in configs.items():
#         result = train_and_evaluate(
#             width_name=name,
#             widths=widths,
#             train_loader=train_loader,
#             test_loader=test_loader,
#             input_size=input_size,
#             num_classes=num_classes,
#             epochs=epochs
#         )
#         results.append(result)

#     logging.info("\nСводка результатов:")
#     for r in results:
#         logging.info(f"{r['width_name']}: acc={r['final_test_acc']:.4f}, "
#                      f"параметров={r['n_params']}, время={r['time']:.1f}с")


# if __name__ == "__main__":
#     """
#     Основная точка входа — запуск эксперимента.
#     """
#     run_width_experiments()



# 2.2 Оптимизация архитектуры

import torch
import itertools
import time
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from datasets import get_mnist_loaders
from models import FullyConnectedModel
from trainer import train_model
from utils import count_parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Настройка логирования:
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    handlers=[
        logging.FileHandler("grid_search.log"),
        logging.StreamHandler()
    ]
)


def build_architecture(widths):
    """
    Создает архитектуру модели по заданным ширинам слоев.

    :param widths: список ширин скрытых слоев
    :return: список конфигураций слоев
    """
    layers = []
    for w in widths:
        layers.append({"type": "linear", "size": w})
        layers.append({"type": "relu"})
    return layers


def train_and_eval_architecture(name, widths, train_loader, test_loader, input_size, num_classes, epochs):
    """
    Обучает и оценивает архитектуру.

    :param name: строковое описание архитектуры
    :param widths: список ширин скрытых слоев
    :return: финальная точность на тесте
    """
    logging.info(f"Архитектура: {name} — ширины {widths}")

    layers = build_architecture(widths)
    model = FullyConnectedModel(input_size=input_size, num_classes=num_classes, layers=layers).to(device)

    start_time = time.time()
    history = train_model(model, train_loader, test_loader, epochs=epochs, device=device)
    elapsed_time = time.time() - start_time

    test_acc = history['test_accs'][-1]
    logging.info(f"{name}: тестовая точность={test_acc:.4f}, время={elapsed_time:.1f} сек")
    logging.info("-" * 30)
    return test_acc


def grid_search_architectures():
    """
    Запускает grid search по схемам изменения ширины.
    """
    batch_size = 64
    epochs = 5
    input_size = 28 * 28
    num_classes = 10

    train_loader, test_loader = get_mnist_loaders(batch_size)

    widths_options = [64, 128, 256, 512]
    schemes = ["Постоянная", "Сужение", "Расширение"]

    results = np.zeros((len(widths_options), len(widths_options)))

    for i, w1 in enumerate(widths_options):
        for j, w3 in enumerate(widths_options):
            # Постоянность
            widths_constant = [w1, w1, w1]
            acc_const = train_and_eval_architecture(
                name=f"Постоянная {w1}",
                widths=widths_constant,
                train_loader=train_loader,
                test_loader=test_loader,
                input_size=input_size,
                num_classes=num_classes,
                epochs=epochs
            )

            # Сужение
            widths_narrow = [max(w1, w3), min(w1, w3), min(w1, w3) // 2]
            acc_narrow = train_and_eval_architecture(
                name=f"Сужение {w1}->{w3}",
                widths=widths_narrow,
                train_loader=train_loader,
                test_loader=test_loader,
                input_size=input_size,
                num_classes=num_classes,
                epochs=epochs
            )

            # Расширение
            widths_wide = [min(w1, w3) // 2, min(w1, w3), max(w1, w3)]
            acc_wide = train_and_eval_architecture(
                name=f"Расширение {w1}->{w3}",
                widths=widths_wide,
                train_loader=train_loader,
                test_loader=test_loader,
                input_size=input_size,
                num_classes=num_classes,
                epochs=epochs
            )

            # сохраняем лучшую из трех
            best_acc = max(acc_const, acc_narrow, acc_wide)
            results[i, j] = best_acc

    plot_heatmap(results, widths_options)


def plot_heatmap(results, widths_options):
    """
    Визуализирует результаты в виде heatmap.

    :param results: матрица точностей
    :param widths_options: список ширин
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(results, annot=True, fmt=".2f", xticklabels=widths_options, yticklabels=widths_options, cmap="Blues")
    plt.title("Heatmap тестовой точности для разных ширин (лучшая схема)")
    plt.xlabel("Ширина 3-го слоя")
    plt.ylabel("Ширина 1-го слоя")
    plt.tight_layout()
    plt.savefig("grid_search_heatmap.png")
    plt.show()


if __name__ == "__main__":
    """
    Основная точка входа — запуск grid search эксперимента.
    """
    grid_search_architectures()

