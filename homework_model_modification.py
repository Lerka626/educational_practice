# 1.1 Расширение линейной регрессии

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import make_regression_data, mse, log_epoch, RegressionDataset


# Добавляю импорты:
import matplotlib.pyplot as plt
import logging
import numpy as np


class LinearRegression(nn.Module):
    """
    Простая линейная регрессионная модель с одним линейным слоем.
    """
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


# Добавляю класс для обучения модели:
def trainLinReg(model, dataloader, criterion, optimizer, epochs=100,
          l1_lambda=0.0, l2_lambda=0.0, patience=10):
    """
    Обучает модель с регуляризацией и ранней остановкой.

    Parameters:
        model: nn.Module — модель
        dataloader: DataLoader — загрузчик данных
        criterion: функция потерь
        optimizer: оптимизатор
        epochs: int — количество эпох
        l1_lambda: float — коэффициент L1 регуляризации
        l2_lambda: float — коэффициент L2 регуляризации
        patience: int — количество эпох без улучшения до early stopping

    Returns:
        losses: List[float] — список потерь по эпохам
    """
    best_loss = float('inf')
    epochs_no_improve = 0
    losses = []

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        model.train()
        
        # Устраняю реализацию через индексацию, тк при ранней остановке работа будет некорректной
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)


            # Добавляю L1 и L2 регуляризацию
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            l2_norm = sum((p ** 2).sum() for p in model.parameters())
            # Формула функции потерь для подсчета = СУММ(лямбда1*L1 + лямбда2*L2)
            loss += l1_lambda * l1_norm + l2_lambda * l2_norm


            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)


        # Добавляю - Быстрая остановка
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'linreg_torch.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info(f'Ранняя остановка сработала на эпохе: {epoch}')
                break

        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss)

    return losses


# Добавляю - визуализация потерь по эпохам
def plot_lossesLinReg(losses):
    """
    Строит график потерь по эпохам.
    """
    plt.plot(losses)
    plt.xlabel("Эпоха")
    plt.ylabel("Средняя потеря")
    plt.title("График потерь по эпохам")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # Добавляю - настраивание логирования: время, уровень и сообщение
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


    # Генерируем данные
    X, y = make_regression_data(n=200)
    
    # Добавляю - проверка на соразмерность
    print(X.ndim == 2 and y.ndim == 2)


    # Создаём датасет и даталоадер
    dataset = RegressionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')
    
    # Создаём модель, функцию потерь и оптимизатор
    model = LinearRegression(in_features=1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # Обучаем модель -- Изменила код, тк функция для обучения уже создана
    losses = trainLinReg(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=100,
        l1_lambda=1e-4,
        l2_lambda=1e-4,
        patience=10
    )
    

    # Добавляю вывод визуализации
    plot_lossesLinReg(losses)


    # Убрала сохранение, то оно есть в функции обучения

    
    # Загружаем модель
    new_model = LinearRegression(in_features=1)
    new_model.load_state_dict(torch.load('linreg_torch.pth'))
    new_model.eval() 



# 1.2 Расширение логистической регрессии

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import make_classification_data, accuracy, log_epoch, ClassificationDataset

# Добавляю импорты:
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
import unittest


class LogisticRegression(nn.Module):
    """
    Многоклассовая логистическая регрессия.
    """
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)


# Добавляю функцию для обучения модели:
def train(model, dataloader, criterion, optimizer, epochs=100):
    """
    Обучение модели логистической регрессии.

    Parameters:
        model: nn.Module — модель логистической регрессии
        dataloader: DataLoader — загрузчик данных
        criterion: функция потерь
        optimizer: оптимизатор
        epochs: int — количество эпох

    Returns:
        all_preds, all_labels — списки предсказаний и истинных значений
    """
    for epoch in range(1, epochs + 1):
        total_loss = 0
        all_preds = []
        all_labels = []

        model.train()
        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Добавляю - сбор предсказаний и истинных меток
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())


        # Изменила подсчет, тк если батчи разного размера, то это снизит точность модели
        avg_loss = total_loss / len(dataloader)
        acc = accuracy(torch.tensor(all_preds), torch.tensor(all_labels))


        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss, acc=acc)

    return all_preds, all_labels


# Дабавляю функцию для визуализации:
def plot_confusion(y_true, y_pred, class_names):
    """
    Визуализация матрицы ошибок.
    """
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


# Добавляю функцию для расчета метрик модели:
def evaluate_metrics(y_true, y_pred, y_probs, num_classes):
    """
    Расчет и вывод метрик.
    """
    logging.info("Метрики по эпохам:")
    logging.info(f"Accuracy:  {accuracy(torch.tensor(y_pred), torch.tensor(y_true)):.4f}")
    logging.info(f"Precision: {precision_score(y_true, y_pred, average='macro'):.4f}")
    logging.info(f"Recall:    {recall_score(y_true, y_pred, average='macro'):.4f}")
    logging.info(f"F1-score:  {f1_score(y_true, y_pred, average='macro'):.4f}")

    # Добаляю: попытка посчитать доп. метрику - ROC-AUC
    try:
        y_true_1h = nn.functional.one_hot(torch.tensor(y_true), num_classes=num_classes).numpy()
        roc_auc = roc_auc_score(y_true_1h, y_probs, multi_class='ovr')
        logging.info(f"ROC-AUC:   {roc_auc:.4f}")
    except Exception as e:
        logging.warning(f"ROC-AUC не рассчитан: {e}")

if __name__ == '__main__':
    # Добавляю логирование:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    # Задаю 3 класса, чтобы модель стала многоклассовой
    num_classes = 3

    # Увеличила количество генерируемых объектов:
    X, y = make_classification_data(n=300)

    # Добавила адаптацию под CrossEntropy (удаление личших размерностей из меток и преобразование их в int64)
    y = y.squeeze().long()

    dataset = ClassificationDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')

    # Добавила количество классов при создании модели
    model = LogisticRegression(in_features=2, num_classes=num_classes)

    # Создаем функцию потрерь и оптимизатор:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Получаю предсказания и истинные метри после обучения модели
    preds, labels = train(model, dataloader, criterion, optimizer, epochs=100)

    # Перевод модели в режим интерфейса и подсчет прогноза вероятностей
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).numpy()

    # Вывод метрик и матрицы ошибок
    evaluate_metrics(labels, preds, probs, num_classes=num_classes)
    plot_confusion(labels, preds, class_names=[f"Class {i}" for i in range(num_classes)])

    torch.save(model.state_dict(), "logreg_torch_multiclass.pth")

    new_model = LogisticRegression(in_features=2, num_classes=num_classes)
    new_model.load_state_dict(torch.load("logreg_torch_multiclass.pth"))
    new_model.eval()

# Создаю класс юнит-теста, в нем функция для проверки, что accuracy возвращает верный результат
class TestUtils(unittest.TestCase):
    def test_accuracy(self):
        y_logits = torch.tensor([0.8, 0.3, 0.9])
        y_true = torch.tensor([1.0, 0.0, 1.0])

        acc = accuracy(y_logits, y_true)
        self.assertAlmostEqual(acc, 1.0)

# Запуск юнит-теста
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

