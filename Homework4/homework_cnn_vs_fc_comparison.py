# 1.1 Сравнение на MNIST

import torch
import torch.nn as nn
import time
import logging
from datasets import get_mnist_loaders
from models import SimpleCNN, CNNWithResidual
from trainer import train_model
from utils import plot_training_history, count_parameters, compare_models
import matplotlib.pyplot as plt

# Настраиваем логирование, работу на процессоре/видеокарте 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Настраиваем данные
train_loader, test_loader = get_mnist_loaders(batch_size=64)

class FullyConnectedNet(nn.Module):
    """
    Полносвязная сеть с 3 скрытыми слоями
    """
    def __init__(self, input_size=28*28, hidden_sizes=[512, 256, 128], num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], num_classes)
        )

    def forward(self, x):
        return self.net(x)

def evaluate_inference_time(model, loader, device):
    """
    Замеряет среднее время инференса на батч
    """
    model.eval()
    total_time = 0
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            start = time.time()
            _ = model(data)
            end = time.time()
            total_time += (end - start)
    return total_time / len(loader)

def experiment(model, name, train_loader, test_loader, epochs=5):
    """
    Проводит эксперимент: обучает модель и собирает метрики
    """
    model = model.to(device)
    logging.info(f"Начало эксперимента: {name}")
    n_params = count_parameters(model)
    logging.info(f"Параметров: {n_params}")

    start_time = time.time()
    history = train_model(model, train_loader, test_loader, epochs=epochs, device=device)
    training_time = time.time() - start_time
    logging.info(f"Время обучения: {training_time:.2f} сек")

    inference_time = evaluate_inference_time(model, test_loader, device)
    logging.info(f"Среднее время инференса на батч: {inference_time:.4f} сек")

    logging.info(f"Финальная test accuracy: {history['test_accs'][-1]:.4f}")

    result = {
        "name": name,
        "model": model,
        "history": history,
        "n_params": n_params,
        "training_time": training_time,
        "inference_time": inference_time
    }
    return result

def plot_comparison(results):
    """
    Визуализирует сравнение моделей
    """
    # Кривые обучения
    for r in results:
        plt.plot(r['history']['test_accs'], label=f"{r['name']}")
    plt.title("Test Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

    # Время обучения
    names = [r['name'] for r in results]
    times = [r['training_time'] for r in results]
    plt.bar(names, times)
    plt.title("Training Time (sec)")
    plt.ylabel("Seconds")
    plt.show()

    # Количество параметров
    params = [r['n_params'] for r in results]
    plt.bar(names, params)
    plt.title("Number of Parameters")
    plt.ylabel("Count")
    plt.show()

    # Время инференса
    inference = [r['inference_time'] for r in results]
    plt.bar(names, inference)
    plt.title("Inference Time per Batch (sec)")
    plt.ylabel("Seconds")
    plt.show()

def main():
    """
    Запускает все эксперименты
    """
    results = []

    fc_model = FullyConnectedNet()
    results.append(experiment(fc_model, "Fully Connected", train_loader, test_loader))

    simple_cnn = SimpleCNN(input_channels=1, num_classes=10)
    results.append(experiment(simple_cnn, "Simple CNN", train_loader, test_loader))

    residual_cnn = CNNWithResidual(input_channels=1, num_classes=10)
    results.append(experiment(residual_cnn, "CNN with Residual", train_loader, test_loader))

    plot_comparison(results)

    compare_models(
        fc_history=results[0]['history'],
        cnn_history=results[1]['history']
    )
    compare_models(
        fc_history=results[0]['history'],
        cnn_history=results[2]['history']
    )

if __name__ == "__main__":
    main()


# 1.2 Сравнение на CIFAR-10

import torch
import torch.nn as nn
import time
import logging
from datasets import get_cifar10_loaders
from trainer import train_model
from utils import count_parameters, plot_training_history
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Настройки логирования, устройства для обучения и данных
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

train_loader, test_loader = get_cifar10_loaders(batch_size=64)


class FullyConnectedCIFAR(nn.Module):
    """
    Глубокая полносвязная сеть для CIFAR-10
    """
    def __init__(self, input_size=32*32*3, hidden_sizes=[1024, 512, 256], num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_sizes[0]), nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]), nn.ReLU(),
            nn.Linear(hidden_sizes[2], num_classes)
        )

    def forward(self, x):
        return self.net(x)


from models import ResidualBlock

class CNNResidualCIFAR(nn.Module):
    """
    CNN с Residual-блоками для CIFAR-10
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.res1 = ResidualBlock(32, 32)
        self.res2 = ResidualBlock(32, 64, 2)
        self.res3 = ResidualBlock(64, 64)
        self.pool = nn.AdaptiveAvgPool2d((4,4))
        self.fc = nn.Linear(64*4*4, num_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CNNResidualRegCIFAR(nn.Module):
    """
    CNN с Residual-блоками и регуляризацией для CIFAR-10
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.res1 = ResidualBlock(32, 32)
        self.dropout1 = nn.Dropout2d(0.2)
        self.res2 = ResidualBlock(32, 64, 2)
        self.dropout2 = nn.Dropout2d(0.2)
        self.res3 = ResidualBlock(64, 64)
        self.pool = nn.AdaptiveAvgPool2d((4,4))
        self.fc = nn.Linear(64*4*4, num_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.dropout1(x)
        x = self.res2(x)
        x = self.dropout2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def experiment(model, name, train_loader, test_loader, epochs=10, l2=0.0):
    """
    Проводит эксперимент и возвращает метрики
    """
    model = model.to(device)
    logging.info(f"--- {name} ---")
    logging.info(f"Параметров: {count_parameters(model)}")

    start = time.time()
    history = train_model(model, train_loader, test_loader, epochs=epochs, device=device, lr=1e-3, weight_decay=l2)
    elapsed = time.time() - start
    logging.info(f"Время обучения: {elapsed:.1f} сек")

    return {
        "name": name,
        "model": model,
        "history": history,
        "training_time": elapsed
    }


def plot_confusion_matrix(model, loader, classes, name):
    """
    Строит confusion matrix
    """
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x).argmax(1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(out.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def plot_gradient_flow(model, loader):
    """
    Визуализирует градиенты
    """
    model.train()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    out = model(x)
    loss = nn.CrossEntropyLoss()(out, y)
    loss.backward()
    ave_grads = []
    layers = []
    for n, p in model.named_parameters():
        if p.requires_grad and "weight" in n:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
    plt.plot(ave_grads, alpha=0.7, marker='o')
    plt.hlines(0, 0, len(ave_grads), lw=2, color="k")
    plt.xticks(range(0,len(ave_grads)), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=0)
    plt.title("Average Gradient Flow")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    """
    Запуск экспериментов
    """
    results = []
    classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

    fc = FullyConnectedCIFAR()
    results.append(experiment(fc, "Fully Connected", train_loader, test_loader))

    cnn_res = CNNResidualCIFAR()
    results.append(experiment(cnn_res, "CNN Residual", train_loader, test_loader))

    cnn_res_reg = CNNResidualRegCIFAR()
    results.append(experiment(cnn_res_reg, "CNN Residual + Reg", train_loader, test_loader, l2=1e-4))

    # Визуализация
    for res in results:
        plot_training_history(res['history'])
        plot_confusion_matrix(res['model'], test_loader, classes, res['name'])
        plot_gradient_flow(res['model'], train_loader)

    # Сравнение по времени
    names = [r['name'] for r in results]
    times = [r['training_time'] for r in results]
    plt.bar(names, times)
    plt.title("Training Time Comparison")
    plt.ylabel("Seconds")
    plt.show()

if __name__ == "__main__":
    main()

