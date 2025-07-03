# 2.1 Кастомный Dataset класс

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class CSVDataset(Dataset):
    """
    Кастомный PyTorch Dataset для чтения и предобработки CSV-файла.
    
    Поддерживает числовые, категориальные и бинарные признаки.
    """
    def __init__(self, filepath, target_column, categorical_cols=None, numerical_cols=None):
        """
        Инициализация датасета.
        
        :param filepath: путь к CSV-файлу
        :param target_column: имя колонки с целевой переменной
        :param categorical_cols: список категориальных признаков
        :param numerical_cols: список числовых признаков
        """
        logging.info(f"Загрузка данных из {filepath}")
        self.data = pd.read_csv(filepath)

        self.target_column = target_column
        self.categorical_cols = categorical_cols or []
        self.numerical_cols = numerical_cols or []

        self._build_pipeline()
        self._preprocess()

    def _build_pipeline(self):
        """Создание пайплайна для обработки признаков."""
        self.pipeline = ColumnTransformer(transformers=[
            ('num', StandardScaler(), self.numerical_cols),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), self.categorical_cols)
        ])

    def _preprocess(self):
        """Предобработка признаков и целевой переменной."""
        logging.info("Применение пайплайна для обработки признаков")
        features = self.data[self.numerical_cols + self.categorical_cols]
        self.X = self.pipeline.fit_transform(features)

        self.y = self.data[self.target_column].values
        if self.y.dtype == object:
            self.y = pd.factorize(self.y)[0]  # категоризация целевой переменной

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        y_tensor = torch.tensor(self.y[idx], dtype=torch.long)
        return x_tensor, y_tensor

    def plot_distributions(self):
        """Визуализация распределения признаков и целевой переменной."""
        logging.info("Визуализация распределений признаков")
        for col in self.numerical_cols:
            sns.histplot(self.data[col], kde=True)
            plt.title(f"Распределение: {col}")
            plt.show()

        sns.countplot(x=self.data[self.target_column])
        plt.title("Распределение классов")
        plt.show()

if __name__ == '__main__':
    dataset = CSVDataset(
        filepath='student_habits_performance.csv',
        target_column='exam_score',
        numerical_cols = [
            'age',
            'study_hours_per_day',
            'social_media_hours',
            'netflix_hours',
            'attendance_percentage',
            'sleep_hours',
            'exercise_frequency',
            'mental_health_rating'
        ],
        categorical_cols = [
            'gender',
            'part_time_job',
            'parental_education_level',
            'internet_quality',
            'diet_quality',
            'extracurricular_participation'
        ]
    )

    print(f"Размер: {len(dataset)}")
    x, y = dataset[0]
    print(f"X shape: {x.shape}, Y: {y}")

    dataset.plot_distributions()


import unittest
import pandas as pd
import torch
import os

class TestCSVDataset(unittest.TestCase):
    def setUp(self):
        # Создаю временный CSV-файл с нужными столбцами
        df = pd.DataFrame({
            'age': [16, 17, 18],
            'study_hours_per_day': [2, 3, 1.5],
            'social_media_hours': [3, 1, 2],
            'netflix_hours': [1, 2, 1],
            'attendance_percentage': [95, 88, 100],
            'sleep_hours': [7, 6, 8],
            'diet_quality': [3, 2, 4],
            'exercise_frequency': [2, 0, 3],
            'mental_health_rating': [4, 3, 5],
            'gender': ['female', 'male', 'female'],
            'part_time_job': ['yes', 'no', 'no'],
            'parental_education_level': ['high', 'low', 'medium'],
            'internet_quality': ['good', 'poor', 'excellent'],
            'extracurricular_participation': ['yes', 'no', 'yes'],
            'exam_score': [85, 75, 90]  # Целевая переменная
        })
        df.to_csv('temp_student_test.csv', index=False)

    def test_loading(self):
        dataset = CSVDataset(
            filepath='temp_student_test.csv',
            target_column='exam_score',
            numerical_cols=[
                'age', 'study_hours_per_day', 'social_media_hours',
                'netflix_hours', 'attendance_percentage', 'sleep_hours',
                'diet_quality', 'exercise_frequency', 'mental_health_rating'
            ],
            categorical_cols=[
                'gender', 'part_time_job', 'parental_education_level',
                'internet_quality', 'extracurricular_participation'
            ]
        )
        self.assertEqual(len(dataset), 3)
        x, y = dataset[0]
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(x.dim(), 1)

    def tearDown(self):
        os.remove('temp_student_test.csv')


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)



# 2.2 Эксперименты с различными датасетами

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from homework_model_modification import LinearRegression, trainLinReg, plot_lossesLinReg 
import torch.nn as nn
import torch.optim as optim

df = pd.read_csv("diabetes_prediction_dataset.csv")

target_col = "diabetes"
df[target_col] = df[target_col].astype(float) 

# Числовые и категориальные признаки
num_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
cat_cols = ["gender", "smoking_history"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

X = preprocessor.fit_transform(df)
y = df[target_col].values.reshape(-1, 1)

# Перевожу в тензоры
X_tensor = torch.tensor(X.astype(np.float32))
y_tensor = torch.tensor(y.astype(np.float32))

class DiabetesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = DiabetesDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = LinearRegression(in_features=X.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

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

plot_lossesLinReg(losses)

model.eval()
with torch.no_grad():
    preds = model(X_tensor)
    preds_binary = (preds > 0.5).float()
    accuracy = (preds_binary == y_tensor).float().mean().item()
    print(f"Точность (accuracy): {accuracy:.4f}")



# ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ С СЕРДЦАМИ
import pandas as pd
import numpy as np

df = pd.read_csv("heart.csv")

print(df.head())
print(df["target"].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Отделяем признаки и целевую переменную
X = df.drop("target", axis=1).values.astype(np.float32)
y = df["target"].values.astype(np.int64)

# Количество классов
num_classes = len(np.unique(y))
print("Классов:", num_classes)

# Нормализация
scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)

# Деление на обучающую и тестовую выборку
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import torch
from torch.utils.data import Dataset, DataLoader

class HeartDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = HeartDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = HeartDataset(X_test, y_test)

from homework_model_modification import LogisticRegression, train, evaluate_metrics, plot_confusion 
import torch.nn as nn
import torch.optim as optim

in_features = X.shape[1]

model = LogisticRegression(in_features=in_features, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

preds, labels = train(model, train_loader, criterion, optimizer, epochs=100)

model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test)
    logits = model(X_test_tensor)
    probs = torch.softmax(logits, dim=1).numpy()
    preds_test = torch.argmax(logits, dim=1).numpy()

evaluate_metrics(y_test, preds_test, probs, num_classes)
plot_confusion(y_test, preds_test, class_names=[f"Class {i}" for i in range(num_classes)])

torch.save(model.state_dict(), "logreg_heart.pth")

# Загрузка модели
new_model = LogisticRegression(in_features=in_features, num_classes=num_classes)
new_model.load_state_dict(torch.load("logreg_heart.pth"))
new_model.eval()
