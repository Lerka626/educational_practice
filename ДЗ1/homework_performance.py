# task 3.1-3.3

import torch
import time
import pandas as pd

def generate_cpu_tensors():
    """
    Создает большие тензоры на CPU.
    """
    return {
        'A_tensor': torch.randn(64, 1024, 1024),
        'B_tensor': torch.randn(128, 512, 512),
        'C_tensor': torch.randn(256, 256, 256),
    }

def measure_cpu_time(operation, tensor):
    """
    Измеряет время выполнения операции на CPU.
    """
    start = time.time()
    operation(tensor)
    end = time.time()
    return (end - start) * 1000  # миллисекунды

def run_cpu_benchmark():
    tensors = generate_cpu_tensors()
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
        t_cpu = measure_cpu_time(op, t)
        results.append({
            "Операция": name,
            "CPU": round(t_cpu, 2),
        })

    return pd.DataFrame(results)

cpu_df = run_cpu_benchmark()
print("CPU Benchmark:")
print(cpu_df.to_markdown(index=False))
