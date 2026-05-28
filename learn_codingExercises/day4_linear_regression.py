"""Day 4: linear regression two ways.

Fit y = 2x - 1 to noisy data. First by hand, then with nn.Linear.
"""

import numpy as np
import torch
import torch.nn as nn

# --- data: 100 points along y = 2x - 1 + a little noise ---
np.random.seed(0)
x = torch.tensor(np.random.uniform(-3, 3, (100, 1)), dtype=torch.float32)
y = 2 * x - 1 + 0.3 * torch.randn_like(x)


# --- Way 1: from scratch. You manage w and b yourself. ---
def from_scratch():
    w = torch.zeros(1, requires_grad=True)   # requires_grad turns on autograd
    b = torch.zeros(1, requires_grad=True)

    for _ in range(200):
        y_pred = w * x + b
        loss = ((y_pred - y) ** 2).mean()    # mean squared error

        loss.backward()                      # fills w.grad and b.grad

        with torch.no_grad():                # update without recording it
            w -= 0.05 * w.grad
            b -= 0.05 * b.grad
            w.grad.zero_()                   # grads accumulate; clear them
            b.grad.zero_()

    print(f"scratch:    w={w.item():.2f}  b={b.item():.2f}")


# --- Way 2: same math, but nn.Linear + optimizer hide the bookkeeping. ---
def with_nn_linear():
    model = nn.Linear(1, 1)                          # holds w (.weight) and b (.bias)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()

    for _ in range(200):
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()                             # replaces the manual w -= lr*grad

    print(f"nn.Linear:  w={model.weight.item():.2f}  b={model.bias.item():.2f}")


def main() -> None:
    from_scratch()
    with_nn_linear()
    print("true:       w=2.00  b=-1.00")

if __name__ == "__main__":
    main()
