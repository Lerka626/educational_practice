# 3.1 Сравнение техник регуляризации

import torch
import time
import logging
import matplotlib.pyplot as plt

from datasets import get_mnist_loaders
from models import FullyConnectedModel
from trainer import train_model
from utils import count_parameters

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_layers_with_regularization(reg_type, dropout_rate=0.5):
    """
    Формирует список слоев модели с заданной регуляризацией.
    """
    layers = [
        {"type": "linear", "size": 512},
        {"type": "relu"},
    ]

    if reg_type == "dropout":
        layers.append({"type": "dropout", "rate": dropout_rate})
    elif reg_type == "batchnorm":
        layers.insert(1, {"type": "batch_norm"})
    elif reg_type == "dropout_batchnorm":
        layers.insert(1, {"type": "batch_norm"})
        layers.append({"type": "dropout", "rate": dropout_rate})

    layers += [
        {"type": "linear", "size": 256},
        {"type": "relu"},
        {"type": "linear", "size": 128},
        {"type": "relu"},
    ]
    return layers


def visualize_weights(model, title):
    """
    Строит гистограмму весов модели.
    """
    all_weights = torch.cat([p.view(-1).detach().cpu() for p in model.parameters() if p.requires_grad])
    plt.hist(all_weights.numpy(), bins=50)
    plt.title(f"Распределение весов: {title}")
    plt.xlabel("Значение")
    plt.ylabel("Частота")
    plt.show()


def train_with_regularization(name, layers, train_loader, test_loader, l2_decay=0.0, epochs=5):
    """
    Обучает модель с заданными слоями и L2-регуляризацией.
    """
    logging.info(f"Начало эксперимента: {name}")
    model = FullyConnectedModel(
        input_size=784,
        num_classes=10,
        layers=layers
    ).to(device)

    logging.info(f"Параметров в модели: {count_parameters(model)}")

    import torch.optim as optim  # импортим здесь для явности
    import torch.nn as nn

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=l2_decay)

    history = {
        'train_losses': [],
        'test_losses': [],
        'train_accs': [],
        'test_accs': []
    }

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_acc = correct / total
        train_loss /= total

        # оценка на тесте
        model.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                x = x.view(x.size(0), -1)
                out = model(x)
                loss = criterion(out, y)

                test_loss += loss.item() * x.size(0)
                _, pred = torch.max(out, 1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        test_acc = correct / total
        test_loss /= total

        history['train_losses'].append(train_loss)
        history['test_losses'].append(test_loss)
        history['train_accs'].append(train_acc)
        history['test_accs'].append(test_acc)

        logging.info(f"[{name}] Эпоха {epoch+1}/{epochs} — Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")

    elapsed = time.time() - start_time

    logging.info(f"Время обучения: {elapsed:.1f} сек")
    logging.info(f"Точность на тесте: {history['test_accs'][-1]:.4f}")
    visualize_weights(model, name)

    return {
        "name": name,
        "test_acc": history['test_accs'][-1],
        "history": history
    }


def plot_histories(results):
    """
    Визуализирует кривые обучения для всех экспериментов.
    """
    plt.figure(figsize=(12, 6))
    for result in results:
        plt.plot(result['history']['test_accs'], label=result['name'])
    plt.title("Точность на тесте по эпохам")
    plt.xlabel("Эпоха")
    plt.ylabel("Точность")
    plt.legend()
    plt.show()


def run_regularization_experiments():
    """
    Основная функция запуска экспериментов.
    """
    train_loader, test_loader = get_mnist_loaders(batch_size=64)

    results = []

    # Без регуляризации
    layers = build_layers_with_regularization(reg_type=None)
    results.append(train_with_regularization("Без регуляризации", layers, train_loader, test_loader))

    # Dropout: 0.1, 0.3, 0.5
    for rate in [0.1, 0.3, 0.5]:
        layers = build_layers_with_regularization(reg_type="dropout", dropout_rate=rate)
        results.append(train_with_regularization(f"Dropout {rate}", layers, train_loader, test_loader))

    # BatchNorm
    layers = build_layers_with_regularization(reg_type="batchnorm")
    results.append(train_with_regularization("BatchNorm", layers, train_loader, test_loader))

    # Dropout + BatchNorm
    layers = build_layers_with_regularization(reg_type="dropout_batchnorm", dropout_rate=0.3)
    results.append(train_with_regularization("Dropout + BatchNorm", layers, train_loader, test_loader))

    # L2 (weight decay)
    layers = build_layers_with_regularization(reg_type=None)
    results.append(train_with_regularization("L2 рег. (0.01)", layers, train_loader, test_loader, l2_decay=0.01))

    plot_histories(results)

    # Анализ
    logging.info("\n=== Результаты ===")
    for r in results:
        logging.info(f"{r['name']}: финальная точность = {r['test_acc']:.4f}")


if __name__ == "__main__":
    run_regularization_experiments()



# 3.2 Адаптивная регуляризация 


