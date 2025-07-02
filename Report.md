Код на GPU (использовала google colab notebook):

# task 3.1–3.3
import torch
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_gpu_tensors():
    """
    Создает большие тензоры на GPU.
    """
    return {
        'A_tensor': torch.randn(64, 1024, 1024, device=device),
        'B_tensor': torch.randn(128, 512, 512, device=device),
        'C_tensor': torch.randn(256, 256, 256, device=device),
    }

def measure_gpu_time(operation, tensor):
    """
    Измеряет время выполнения операции на GPU с помощью torch.cuda.Event().
    """
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    operation(tensor)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)

def run_gpu_benchmark():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна")

    tensors = generate_gpu_tensors()
    t = tensors['C_tensor']

    operations = {
        "Матричное умножение": lambda x: torch.matmul(x, x.transpose(-1, -2)),
        "Сложение": lambda x: x + x,
        "Умножение": lambda x: x * x,
        "Транспонирование": lambda x: x.transpose(-1, -2),
        "Суммирование": lambda x: x.sum()
    }

    results = []
    for name, op in operations.items():
        t_gpu = measure_gpu_time(op, t)
        results.append({
            "Операция": name,
            "GPU": round(t_gpu, 2),
        })

    return pd.DataFrame(results)

gpu_df = run_gpu_benchmark()
print("GPU Benchmark:")
print(gpu_df.to_markdown(index=False))


Результат:


GPU Benchmark:
| Операция            |    GPU |
|:--------------------|-------:|
| Матричное умножение | 187.33 |
| Сложение            |  54.02 |
| Умножение           |  34.4  |
| Транспонирование    |   0.07 |
| Суммирование        |  41.49 |


Код на CPU в файле performance
Результат:

CPU Benchmark:
| Операция            |   CPU |
|:--------------------|------:|
| Матричное умножение | 96.48 |
| Сложение            |  9.56 |
| Умножение           |  8.52 |
| Транспонирование    |  0    |
| Суммирование        |  3.02 |


Какие операции получают наибольшее ускорение на GPU?

Если честно — никакие особо не ускорились
Обычно GPU должен быть быстрее, но в моем случае:
-Матричное умножение на GPU заняло 187.33 мс, а на CPU — 96.48 мс
-Поэлементные операции (+, *) на GPU тоже медленнее, чем на CPU

Вывод: у нас GPU работал медленнее, чем CPU почти везде. Это странно..


Почему некоторые операции могут быть медленнее на GPU?

GPU — быстрее только когда:
-данные большие
-операции реально тяжёлые
-и всё уже лежит в видеопамяти
Еще GPU может долго "разогреваться" (запускать ядра и т.п.)
Вывод: простые операции лучше делать на CPU


Как размер матриц влияет на ускорение?

Чем больше матрица, тем больше шансов, что GPU себя покажет, так как он лучше справляется с большим объемом данных
На маленьких матрицах — CPU просто делает всё сразу и быстро


Что происходит при передаче данных между CPU и GPU?

Вот что реально тормозит всё:
-если мы делаем .to("cuda"), PyTorch копирует данные в видеопамять (это медленно)
-если обратно - тоже самое, только в другую сторону
-эта передача может быть медленнее самой операции

Вывод: если гонять данные между процессором и видеокартой — GPU не успевает "разогнаться" и вообще становится бессмысленным 
