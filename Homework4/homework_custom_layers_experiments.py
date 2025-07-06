# 3.1 Реализация кастомных слоев 

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')


class CustomConv2d(nn.Module):
    """
    Кастомный сверточный слой с "channel-wise scaling"
    (дополнительные обучаемые коэффициенты на каждый канал).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = nn.Parameter(torch.ones(out_channels))

    def forward(self, x):
        out = self.conv(x)
        out = out * self.scale.view(1, -1, 1, 1)
        return out


class SpatialAttention(nn.Module):
    """
    Attention механизм: spatial attention map.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attn = torch.sigmoid(self.conv(concat))
        return x * attn


class Swish(nn.Module):
    """
    Кастомная функция активации: Swish.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class CustomPooling(nn.Module):
    """
    Кастомный pooling слой: Generalized mean pooling.
    """
    def __init__(self, p=3):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))

    def forward(self, x):
        x = x.clamp(min=1e-6).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.pow(1./self.p)
        return x


def test_layer(layer, input_shape):
    """
    Тестирует слой на случайном тензоре указанной формы.
    """
    x = torch.randn(*input_shape, requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    logging.info(f"Тест {layer.__class__.__name__}: output shape {y.shape}, grad OK")


def visualize_comparison(layer_std, layer_custom, input_shape, title):
    """
    Визуализирует разницу между стандартным и кастомным слоем.
    """
    x = torch.randn(*input_shape)
    out_std = layer_std(x).detach()
    out_custom = layer_custom(x).detach()

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(out_std[0,0].numpy(), cmap='viridis')
    plt.title("Standard")
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.imshow(out_custom[0,0].numpy(), cmap='viridis')
    plt.title("Custom")
    plt.colorbar()

    plt.suptitle(title)
    plt.show()


def main():
    """
    Точка входа: тестирует все кастомные слои.
    """
    logging.info("Тестирование кастомных слоёв")

    # 1. Custom Conv2d
    standard_conv = nn.Conv2d(3, 16, 3, padding=1)
    custom_conv = CustomConv2d(3, 16, 3, padding=1)
    test_layer(custom_conv, (2,3,32,32))
    visualize_comparison(standard_conv, custom_conv, (1,3,32,32), "Conv2d vs CustomConv2d")

    # 2. Attention
    attention = SpatialAttention()
    test_layer(attention, (2,16,32,32))

    # 3. Swish activation
    swish = Swish()
    test_layer(swish, (2,16,32,32))

    # 4. Custom pooling
    standard_pool = nn.AdaptiveAvgPool2d((1,1))
    custom_pool = CustomPooling()
    test_layer(custom_pool, (2,16,8,8))
    visualize_comparison(standard_pool, custom_pool, (1,16,8,8), "AvgPool vs CustomPooling")

    logging.info("Все кастомные слои протестированы успешно")


if __name__ == "__main__":
    main()



# 3.2 Эксперименты с Residual блоками

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicBlock(nn.Module):
    """Basic residual block with two 3x3 conv layers"""
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


class BottleneckBlock(nn.Module):
    """Bottleneck residual block with 1x1 -> 3x3 -> 1x1"""
    def __init__(self, channels, bottleneck_channels=None):
        super().__init__()
        if bottleneck_channels is None:
            bottleneck_channels = channels // 4
        self.conv1 = nn.Conv2d(channels, bottleneck_channels, 1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, channels, 1)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        return F.relu(out)


class WideBlock(nn.Module):
    """Wide residual block with more channels"""
    def __init__(self, channels, widen_factor=2):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels * widen_factor, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels * widen_factor)
        self.conv2 = nn.Conv2d(channels * widen_factor, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class ResNetVariant(nn.Module):
    def __init__(self, block, num_blocks=3, num_classes=10, channels=16):
        super().__init__()
        self.conv = nn.Conv2d(3, channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.blocks = nn.Sequential(*[block(channels) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.blocks(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)


def experiment(block, name, epochs=3):
    """
    Запускает эксперимент с заданным блоком, логирует параметры и визуализирует обучение
    """
    logging.info(f"Начало эксперимента: {name}")
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    model = ResNetVariant(block).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"{name}: {param_count:,} trainable parameters")

    train_losses, test_accuracies = [], []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for xb, yb in trainloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(trainloader))

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in testloader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        acc = correct / total
        test_accuracies.append(acc)
        logging.info(f"{name} | Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | Test Acc: {acc:.4f}")

    elapsed = time.time() - start_time
    logging.info(f"{name}: обучение завершено за {elapsed:.1f} сек")

    plt.plot(train_losses, label=f"{name} Train Loss")
    plt.plot(test_accuracies, label=f"{name} Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.title(f"Обучение: {name}")
    plt.show()


def main():
    """
    Главная точка входа — запускает все эксперименты
    """
    logging.info("Эксперименты с Residual блоками")
    experiment(BasicBlock, "BasicBlock")
    experiment(BottleneckBlock, "BottleneckBlock")
    experiment(WideBlock, "WideBlock")
    logging.info("Все эксперименты завершены")


if __name__ == "__main__":
    main()

