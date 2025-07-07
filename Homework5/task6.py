import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt
from pathlib import Path

from datasets import CustomImageDataset

train_dir = Path("C:/Users/Lerik/OneDrive/Desktop/all_practices/Practice5/data/train")
val_dir = Path("C:/Users/Lerik/OneDrive/Desktop/all_practices/Practice5/data/val")
result_dir = Path("C:/Users/Lerik/OneDrive/Desktop/all_practices/results")
result_dir.mkdir(exist_ok=True)
result_graph_dir = Path("C:/Users/Lerik/OneDrive/Desktop/all_practices/plots")
result_graph_dir.mkdir(exist_ok=True)

BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 1e-3
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_dataloaders(batch_size: int, img_size: int):
    """
    Создает train и val DataLoader'ы с преобразованиями.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    train_dataset = CustomImageDataset(str(train_dir), transform=transform)
    val_dataset = CustomImageDataset(str(val_dir), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    class_names = train_dataset.get_class_names()
    print(f"Классы: {class_names}")
    return train_loader, val_loader, len(class_names)


def build_model(num_classes: int):
    """
    Загружает предобученную ResNet18 и заменяет последний слой.
    """
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


def train_one_epoch(model, dataloader, optimizer, loss_fn):
    """
    Одна эпоха обучения.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, loss_fn):
    """
    Оценка на валидации.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            outputs = model(x)
            loss = loss_fn(outputs, y)

            running_loss += loss.item() * x.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def plot_curves(train_values, val_values, ylabel, title, filename):
    """
    Строит и сохраняет график по эпохам.
    """
    plt.figure()
    plt.plot(train_values, label="train")
    plt.plot(val_values, label="val")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.savefig(result_graph_dir / filename)
    plt.close()
    print(f"Сохранено: {filename}")


def main():
    """
    Основной пайплайн обучения.
    """
    train_loader, val_loader, num_classes = prepare_dataloaders(BATCH_SIZE, IMG_SIZE)

    model = build_model(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(NUM_EPOCHS):
        print(f"Эпоха {epoch+1}/{NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

    # Визуализация
    plot_curves(train_losses, val_losses,
                ylabel="Loss", title="Loss по эпохам",
                filename="loss_curves.png")

    plot_curves(train_accs, val_accs,
                ylabel="Accuracy", title="Accuracy по эпохам",
                filename="accuracy_curves.png")

    torch.save(model.state_dict(), result_dir / "finetuned_resnet18.pth")
    print("Обучение завершено. Модель и графики сохранены в results/.")


if __name__ == "__main__":
    main()

