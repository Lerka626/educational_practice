# 1.1 Сравнение моделей разной глубины

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time


def make_classification_data(n_samples=1000, n_features=20, n_classes=3, test_size=0.2):
    """
    Генерирует синтетические данные для задачи классификации и возвращает train/test выборки.

    Args:
        n_samples (int): количество образцов
        n_features (int): количество признаков
        n_classes (int): количество классов
        test_size (float): доля тестовой выборки

    Returns:
        X_train (Tensor): обучающие признаки
        y_train (Tensor): обучающие метки
        X_test (Tensor): тестовые признаки
        y_test (Tensor): тестовые метки
        input_size (int): размерность признаков
        num_classes (int): количество классов
    """

    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=10, n_classes=n_classes, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long), \
           torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long), X.shape[1], n_classes


def build_model(input_size, num_classes, num_hidden_layers=0, hidden_size=64):
    """
    Строит полносвязную нейросеть с указанным количеством скрытых слоев.

    Args:
        input_size (int): размерность входных данных
        num_classes (int): количество выходных классов
        num_hidden_layers (int): количество скрытых слоев
        hidden_size (int): размерность скрытых слоев

    Returns:
        nn.Sequential: модель нейросети
    """

    layers = []
    prev_size = input_size
    for i in range(num_hidden_layers):
        layers.append(nn.Linear(prev_size, hidden_size))
        layers.append(nn.ReLU())
        prev_size = hidden_size
    layers.append(nn.Linear(prev_size, num_classes))
    return nn.Sequential(*layers)


def train_and_evaluate(model, train_loader, test_loader, device, epochs=20, lr=0.01):
    """
    Обучает модель и возвращает историю обучения.

    Args:
        model (nn.Module): модель для обучения
        train_loader (DataLoader): загрузчик обучающих данных
        test_loader (DataLoader): загрузчик тестовых данных
        device (torch.device): устройство для вычислений
        epochs (int): количество эпох
        lr (float): скорость обучения

    Returns:
        dict: история обучения (loss и accuracy на train/test)
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, total_train = 0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            preds = outputs.argmax(1)
            train_correct += (preds == y_batch).sum().item()
            total_train += y_batch.size(0)

        model.eval()
        test_loss, test_correct, total_test = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                test_loss += loss.item() * X_batch.size(0)
                preds = outputs.argmax(1)
                test_correct += (preds == y_batch).sum().item()
                total_test += y_batch.size(0)

        history['train_loss'].append(train_loss / total_train)
        history['test_loss'].append(test_loss / total_test)
        history['train_acc'].append(train_correct / total_train)
        history['test_acc'].append(test_correct / total_test)

    return history


def plot_history(history, num_layers):
    """
    Строит графики изменения loss и accuracy для train/test.

    Args:
        history (dict): история обучения
        num_layers (int): количество слоев в модели
    """

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['test_loss'], label='Test Loss')
    plt.title(f'Loss (layers={num_layers})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['test_acc'], label='Test Acc')
    plt.title(f'Accuracy (layers={num_layers})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def run_experiment(num_layers, X_train, y_train, X_test, y_test, input_size, num_classes, device):
    """
    Проводит эксперимент: обучает и оценивает модель с заданным числом слоев.

    Args:
        num_layers (int): общее количество слоев (включая выходной)
        X_train, y_train, X_test, y_test: данные
        input_size (int): размерность признаков
        num_classes (int): количество классов
        device (torch.device): устройство

    Prints:
        Время обучения, финальные метрики
    """

    print(f"\nRunning experiment with {num_layers} layers...")
    model = build_model(input_size, num_classes, num_hidden_layers=num_layers - 1)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    start_time = time.time()
    history = train_and_evaluate(model, train_loader, test_loader, device)
    elapsed = time.time() - start_time

    print(f"Training time for {num_layers} layers: {elapsed:.2f} seconds")
    print(f"Final Train Accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Final Test Accuracy: {history['test_acc'][-1]:.4f}")

    plot_history(history, num_layers)


def main():
    """
    Проводит эксперимент: обучает и оценивает модель с заданным числом слоев.

    Args:
        num_layers (int): общее количество слоев (включая выходной)
        X_train, y_train, X_test, y_test: данные
        input_size (int): размерность признаков
        num_classes (int): количество классов
        device (torch.device): устройство

    Prints:
        Время обучения, финальные метрики
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train, X_test, y_test, input_size, num_classes = make_classification_data(
        n_samples=2000, n_features=20, n_classes=3
    )

    for num_layers in [1, 2, 3, 5, 7]:
        run_experiment(num_layers, X_train, y_train, X_test, y_test, input_size, num_classes, device)


if __name__ == "__main__":
    main()



# 1.2 Анализ переобучения

import torch
from datasets import get_mnist_loaders, get_cifar_loaders
from models import FullyConnectedModel
from trainer import train_model
from utils import plot_training_history

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_layers(depth, use_regularization=False):
    """
    Создает список слоев для модели.

    depth - количество слоев (включая выходной)
    use_regularization - добавлять Dropout и BatchNorm
    """
    layers = []
    hidden_size = 512
    for i in range(depth - 1):
        layers.append({"type": "linear", "size": hidden_size})
        if use_regularization:
            layers.append({"type": "batch_norm"})
        layers.append({"type": "relu"})
        if use_regularization:
            layers.append({"type": "dropout", "rate": 0.3})
    return layers

def run_experiment_for_dataset(name, train_loader, test_loader, input_size, num_classes, depths, epochs):
    """
    Тренирует модели с разной глубиной и регуляризацией,
    строит графики и выводит логи.

    name - название датасета
    """
    results = {}
    print(f"\n=== {name} ===")

    for depth in depths:
        print(f"\nГлубина {depth}, без регуляции")
        layers = build_layers(depth, use_regularization=False)
        model = FullyConnectedModel(input_size=input_size, num_classes=num_classes, layers=layers).to(device)
        history = train_model(model, train_loader, test_loader, epochs=epochs, device=device)
        results[f'{depth}_no_reg'] = history
        plot_training_history(history)

        print(f"\nГлубина {depth}, с Dropout+BatchNorm")
        layers_reg = build_layers(depth, use_regularization=True)
        model_reg = FullyConnectedModel(input_size=input_size, num_classes=num_classes, layers=layers_reg).to(device)
        history_reg = train_model(model_reg, train_loader, test_loader, epochs=epochs, device=device)
        results[f'{depth}_reg'] = history_reg
        plot_training_history(history_reg)

    print("\nАнализ переобучения:")
    for depth in depths:
        train_acc = results[f'{depth}_no_reg']['train_accs']
        test_acc = results[f'{depth}_no_reg']['test_accs']
        for epoch, (tr, ts) in enumerate(zip(train_acc, test_acc), 1):
            if tr - ts > 0.1:
                print(f"Переобучение: глубина {depth}, эпоха {epoch} (train={tr:.3f}, test={ts:.3f})")
                break

    # Оптимальная глубина без рег.
    best_acc = 0
    best_depth = None
    for depth in depths:
        acc = results[f'{depth}_no_reg']['test_accs'][-1]
        print(f"Глубина {depth} итог тест acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_depth = depth
    print(f"Лучший depth без рег: {best_depth} (acc={best_acc:.4f})")

    # Оптимальная глубина с рег.
    best_acc_reg = 0
    best_depth_reg = None
    for depth in depths:
        acc = results[f'{depth}_reg']['test_accs'][-1]
        print(f"Глубина {depth} с рег итог acc: {acc:.4f}")
        if acc > best_acc_reg:
            best_acc_reg = acc
            best_depth_reg = depth
    print(f"Лучший depth с рег: {best_depth_reg} (acc={best_acc_reg:.4f})")

    return results

def main():
    """
    Запускает эксперименты на MNIST и CIFAR10
    """
    batch_size = 64
    epochs = 20
    depths = [1, 2, 3, 5, 7]

    mnist_train_loader, mnist_test_loader = get_mnist_loaders(batch_size)
    cifar_train_loader, cifar_test_loader = get_cifar_loaders(batch_size)

    run_experiment_for_dataset(
        'MNIST',
        mnist_train_loader,
        mnist_test_loader,
        input_size=28*28,
        num_classes=10,
        depths=depths,
        epochs=epochs
    )

    run_experiment_for_dataset(
        'CIFAR10',
        cifar_train_loader,
        cifar_test_loader,
        input_size=32*32*3,
        num_classes=10,
        depths=depths,
        epochs=epochs
    )

if __name__ == "__main__":
    main()

