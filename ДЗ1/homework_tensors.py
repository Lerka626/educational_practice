# task 1.1

import torch

tensor1 = torch.rand(3, 4)
print(tensor1)

tensor2 = torch.zeros(2, 3, 4)
print(tensor2)

tensor3 = torch.ones(5, 5)
print(tensor3)

tensor4 = torch.arange(16).reshape(4, 4)
print(tensor4)


# task 1.2

tensorA = torch.rand(3, 4)
tensorB = torch.rand(4, 3)

tensorA_T = tensorA.transpose(0, 1) # ИЛИ: ..A_T = tensorA.T
print(tensorA)
print(tensorA_T)

multiplicationAB = torch.matmul(tensorA, tensorB)  # ИЛИ: ..AB = A.mm(B); ..AB = A @ B
print(multiplicationAB)

tensorB_T = tensorB.transpose(0, 1)
element_multip_AB = tensorA * tensorB_T
print(tensorB)
print(tensorB_T)
print(element_multip_AB)

A_sum = tensorA.sum()
print(A_sum)


# task 1.3

tensor = torch.rand(5, 5, 5)

print(tensor)
print(tensor[:, 0, :])  # - Первую строку
print(tensor[:, :, -1])  # - Последний столбец
print(tensor[2, 2:4, 2:4])  # - Подматрицу размером 2x2 из центра тензора
print(tensor[::2, ::2, ::2])  # - Все элементы с четными индексами

# task 1.4

tensor_form = torch.arange(24)

print(tensor_form)
print(tensor_form.reshape(2, 12))  # - 2x12
print(tensor_form.reshape(3, 8))  # - 3x8
print(tensor_form.view(4, 6))  # - 4x6
print(tensor_form.view(2, 3, 4))  # - 2x3x4
print(tensor_form.reshape(2, 2, 2, 3))  # - 2x2x2x3
