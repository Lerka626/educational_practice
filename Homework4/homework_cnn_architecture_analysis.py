# 2.1 Влияние размера ядра свертки

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from trainer import train_model
from utils import count_parameters, plot_training_history

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_cifar_loaders(batch_size=64):
    """
    Загружает датасет CIFAR-10 и возвращает DataLoader-ы.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


class CNNKernel3x3(nn.Module):
    """
    CNN с ядрами 3x3, 32 и 64 канала.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class CNNKernel5x5(nn.Module):
    """
    CNN с ядрами 5x5, уменьшим число каналов чтобы сохранить кол-во параметров.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class CNNKernel7x7(nn.Module):
    """
    CNN с ядрами 7x7, ещё меньше каналов.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=7, padding=3)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class CNNKernelCombo1x1_3x3(nn.Module):
    """
    CNN c комбинацией 1x1 + 3x3.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def visualize_first_layer_activations(model, data_loader, device, title=""):
    """
    Визуализирует карты активаций первого слоя выбранной модели на одном изображении из датасета.
    """
    model.eval()
    data_iter = iter(data_loader)
    images, _ = next(data_iter)
    images = images.to(device)

    with torch.no_grad():
        x = model.conv1(images)

    fig, axs = plt.subplots(4, 8, figsize=(12,6))
    axs = axs.flatten()
    for i in range(min(len(axs), x.shape[1])):
        axs[i].imshow(x[0,i].cpu().numpy(), cmap='viridis')
        axs[i].axis('off')
    plt.suptitle(f"First layer activations: {title}")
    plt.show()


def experiment(model_class, name, train_loader, test_loader, epochs, device):
    """
    Запускает эксперимент с заданной моделью: обучение, визуализация, логирование.
    """
    logging.info(f"Начало эксперимента: {name}")
    model = model_class().to(device)
    logging.info(f"Модель {name}, параметров: {count_parameters(model)}")

    start = time.time()
    history = train_model(model, train_loader, test_loader, epochs=epochs, device=device)
    elapsed = time.time() - start
    logging.info(f"Время обучения {name}: {elapsed:.1f} сек")

    plot_training_history(history)
    visualize_first_layer_activations(model, test_loader, device, title=name)

    return history


def main():
    """
    Запускает все эксперименты с разными ядрами свёртки.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Используемое устройство: {device}")

    train_loader, test_loader = get_cifar_loaders(batch_size=64)
    epochs = 5

    results = {}
    results['3x3'] = experiment(CNNKernel3x3, "3x3 kernels", train_loader, test_loader, epochs, device)
    results['5x5'] = experiment(CNNKernel5x5, "5x5 kernels", train_loader, test_loader, epochs, device)
    results['7x7'] = experiment(CNNKernel7x7, "7x7 kernels", train_loader, test_loader, epochs, device)
    results['combo'] = experiment(CNNKernelCombo1x1_3x3, "1x1 + 3x3 combo", train_loader, test_loader, epochs, device)

    # Визуализация итогов
    test_accs = [
        results[k]['test_accs'][-1] for k in results
    ]
    labels = list(results.keys())

    plt.bar(labels, test_accs)
    plt.ylabel("Test Accuracy")
    plt.title("Сравнение точности на CIFAR-10")
    plt.show()


if __name__ == "__main__":
    main()



# 2.2 Влияние глубины CNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
import time
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ShallowCNN(nn.Module):
    """
    Неглубокая CNN: 2 сверточных слоя
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(64*8*8, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class MediumCNN(nn.Module):
    """
    Средняя CNN: 4 сверточных слоя
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(128*8*8, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class DeepCNN(nn.Module):
    """
    Глубокая CNN: 6 сверточных слоев
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(256*8*8, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNetBlock(nn.Module):
    """
    Residual блок
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class ResNetCNN(nn.Module):
    """
    CNN с residual связями
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_layers = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1))
        self.block1 = ResNetBlock(64)
        self.block2 = ResNetBlock(64)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.classifier = nn.Linear(64*8*8, num_classes)

    def forward(self, x):
        x = F.relu(self.conv_layers(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    return running_loss / len(train_loader), correct / total


def test(model, test_loader, criterion):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return loss / len(test_loader), correct / len(test_loader.dataset)


def experiment(model_class, name, epochs=5):
    logging.info(f"==> {name}")
    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    grad_norms = []

    start = time.time()
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        test_loss, test_acc = test(model, test_loader, criterion)

        # измеряем норму градиента первого сверточного слоя
        grad_norm = model.conv_layers[0].weight.grad.norm().item()
        grad_norms.append(grad_norm)

        logging.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} Test Loss={test_loss:.4f} Acc={test_acc:.4f}")
    elapsed = time.time() - start
    logging.info(f"{name} обучена за {elapsed:.1f} секунд")

    # визуализация градиентов
    plt.plot(grad_norms)
    plt.title(f"Gradient norms: {name}")
    plt.xlabel("Epoch")
    plt.ylabel("Norm")
    plt.show()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128)


if __name__ == "__main__":
    experiment(ShallowCNN, "Shallow CNN", epochs=5)
    experiment(MediumCNN, "Medium CNN", epochs=5)
    experiment(DeepCNN, "Deep CNN", epochs=5)
    experiment(ResNetCNN, "ResNet CNN", epochs=5)
