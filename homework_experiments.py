# 3.1 Исследование гиперпараметров

from homework_model_modification import LogisticRegression, train, evaluate_metrics, plot_confusion 
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging


def load_heart_data_from_csv(file_path='heart.csv'):
    """Предобработка и загрузка данных."""
    df = pd.read_csv(file_path)

    X = df.drop(columns=['target']).values
    y = df['target'].values

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    return X, y, le.classes_


def run_experiments(X, y, num_classes, learning_rates, batch_sizes, optimizers_dict, epochs=50):
    """Функция для обучения с учетом скорости и батча."""
    results = []

    for opt_name, opt_func in optimizers_dict.items():
        for lr in learning_rates:
            for batch_size in batch_sizes:
                # Разбиваю данные на train/test 
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y)

                train_ds = TensorDataset(X_train, y_train)
                train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

                model = LogisticRegression(in_features=X.shape[1], num_classes=num_classes)
                criterion = nn.CrossEntropyLoss()
                optimizer = opt_func(model.parameters(), lr=lr)

                preds, labels = train(model, train_dl, criterion, optimizer, epochs=epochs)

                model.eval()
                with torch.no_grad():
                    logits_val = model(X_val)
                    probs_val = torch.softmax(logits_val, dim=1).numpy()
                    preds_val = torch.argmax(logits_val, dim=1).numpy()
                    labels_val = y_val.numpy()

                # Собираю метрики на валидации
                logging.info(f"Optimizer: {opt_name}, LR: {lr}, Batch size: {batch_size}")

                evaluate_metrics(labels_val, preds_val, probs_val, num_classes)

                acc_val = np.mean(preds_val == labels_val)
                results.append({
                    'optimizer': opt_name,
                    'learning_rate': lr,
                    'batch_size': batch_size,
                    'accuracy_val': acc_val,
                })

    return results


def plot_results(results):
    """Визуализация точности в зависимости от learning rate для каждого оптимизатора (с разными batch size)."""
    import pandas as pd
    df = pd.DataFrame(results)
    print(df)

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='learning_rate', y='accuracy_val',
                 hue='optimizer', style='batch_size', markers=True, dashes=False)
    plt.title("Validation Accuracy for Different Hyperparameters")
    plt.xlabel("Learning Rate")
    plt.ylabel("Validation Accuracy")
    plt.xscale('log')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    # Загружаю и подготавливаю данные
    X, y, class_names = load_heart_data_from_csv('heart.csv')
    num_classes = len(class_names)

    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [16, 32, 64]
    optimizers_dict = {
        'SGD': optim.SGD,
        'Adam': optim.Adam,
        'RMSprop': optim.RMSprop
    }

    results = run_experiments(X, y, num_classes, learning_rates, batch_sizes, optimizers_dict, epochs=50)
    plot_results(results)



# 3.2 Feature Engineering

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
import numpy as np
import logging

from homework_model_modification import LogisticRegression, train, evaluate_metrics

def load_and_prepare_data(path='heart.csv'):
    """Загрузка и предобработка данных."""
    df = pd.read_csv(path)
    X = df.drop(columns=['target']).values
    y = df['target'].values

    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y, le.classes_

def add_features(X):
    """Добавляет полиномиальные признаки 2-й степени (включая взаимодействия) и
    статистические признаки по строкам (для каждого объекта), возвращает объединенный массив признаков."""
    # Полиномиальные признаки
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Статистические признаки
    mean_feat = X.mean(axis=1).reshape(-1, 1)
    var_feat = X.var(axis=1).reshape(-1, 1)

    # Объединение
    X_enhanced = np.hstack([X_poly, mean_feat, var_feat])

    return X_enhanced

def scale_data(X_train, X_val):
    """Стандартизация признаков (масштабирование)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled

def train_and_evaluate(X, y, epochs=50, batch_size=32, lr=0.01):
    """Обучение и вывод метрик."""
    num_classes = len(np.unique(y))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    X_train, X_val = scale_data(X_train, X_val)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = LogisticRegression(in_features=X_train.shape[1], num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    train(model, train_dl, criterion, optimizer, epochs=epochs)

    model.eval()
    with torch.no_grad():
        logits_val = model(X_val_t)
        probs_val = torch.softmax(logits_val, dim=1).numpy()
        preds_val = torch.argmax(logits_val, dim=1).numpy()

    print("Метрики на валидации:")
    evaluate_metrics(y_val, preds_val, probs_val, num_classes=num_classes)

    return model, preds_val, y_val


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    # Загрузка исходных данных
    X_base, y, classes = load_and_prepare_data('heart.csv')

    print("Базовая модель:")
    model_base, preds_base, y_val_base = train_and_evaluate(X_base, y)

    # Создание расширенного набора признаков
    X_enhanced = add_features(X_base)

    print("\nМодель с расширенными признаками:")
    model_enhanced, preds_enhanced, y_val_enhanced = train_and_evaluate(X_enhanced, y)
