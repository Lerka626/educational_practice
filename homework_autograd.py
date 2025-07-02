# task 2.1
import torch

x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = torch.tensor(3.0, requires_grad=True)

f = x**2 + y**2 + z**2 + 2*x*y*z

f.backward()

print(x.grad.item())  # df/dx = 2x + 2yz
print(y.grad.item())  # df/dy = 2y + 2xz
print(z.grad.item())  # df/dz = 2z + 2xy


# task 2.2

x = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([2.0, 4.0, 6.0])

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# Линейная модель: y_pred = w * x + b
y_pred = w * x + b

# MSE = (1/n) * СУММ(y_pred - y_true)^2
loss = torch.mean((y_pred - y_true)**2)

loss.backward()

print(w.grad.item())  # ddMSE/dw
print(b.grad.item())  # dMSE/db


# task 2.3

import math

x = torch.tensor(1.0, requires_grad=True)

# f(x) = sin(x^2 + 1)
f = torch.sin(x**2 + 1)

f.backward()

print(x.grad.item())  # Градиент df/dx
