import torch
import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def plot_func():
    x = np.arange(-6, 6, 0.1)
    y = np.arange(-6, 6, 0.1)
    print('x, y range: ', x.shape, y.shape)
    X, Y = np.meshgrid(x, y)
    print('X, Y maps: ', X.shape, Y.shape)
    Z = func([X, Y])

    fig = plt.figure('func')
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.view_init(60, -30)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


def tracker_runner():
    x = torch.tensor([5., 7.], requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=0.001)
    for step in range(20000):
        pred = func(x)
        optimizer.zero_grad()
        # 反向传播，计算出每个参数的梯度
        pred.backward()
        # 根据前面计算出的梯度更新 x
        optimizer.step()

        if step % 2000 == 0:
            print('step {0}: x = {1}, f(x) = {2}'.format(step, x.tolist(), pred.item()))


if __name__ == "__main__":
    plot_func()
    tracker_runner()