import torch
import torchvision
from torch import nn
from torch.nn import functional
from torch import optim
from utils import one_hot, plot_curve, plot_image

# 使用 pytorch 搭建识别手写数字三层网络

batch_size = 512

# noinspection PyAttributeOutsideInit
class MinstNet(nn.Module):

    def __init__(self):
        super(MinstNet, self).__init__()
        # 加载数据
        self.load_data()
        # h1 = w1 * x + b1, w1 参数矩阵的 shape 为 [784, 256]
        self.fc1 = nn.Linear(28 * 28, 256)
        # h2 = w2 * h1 + b2, w2 参数矩阵的 shape 为 [256, 64]
        self.fc2 = nn.Linear(256, 64)
        # h3 = w3 * h2 + b3, w3 参数矩阵的 shape 为 [64, 10]
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # 输入的 x 的 shape 为 [n, 1, 28, 28]，n 表示的就是输入的图像的个数
        # h1 = relu(w1 * x + b1)
        x = functional.relu(self.fc1(x))
        # h2 = relu(w2 * h1 + b2)
        x = functional.relu(self.fc2(x))
        # h3 = w3 * h2 + b3
        x = self.fc3(x)

        return x

    def load_data(self):
        # 1.load data
        self.train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('minist_data/', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])), batch_size=batch_size, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('minist_data/', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])), batch_size=batch_size, shuffle=True)

        # x 表示的是读取到的图片，shape 一般为 [n, 1, 28, 28]，其中 n 表示的是读取到的图片的数目，这里等于上面设置的 batch_size，也就是 512，1 表示的是读入的图片的通道数目，
        # 而 28 * 28 是图片的 width 和 height
        # y 表示的是每个图片对应的 groundtruth， 也等于 batch_size 的大小
        x, y = next(iter(self.train_loader))
        print(x.shape, y.shape, x.min(), x.max())

    def train_runner(self):
        optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        train_loss = []

        # 进行 10 次迭代，每一次迭代都将 MNIST 数据集中的全部数据导入到前面定义的 MinistNet 模型中进行训练
        for epoch in range(10):
            for iter_idx, (x, y) in enumerate(self.train_loader):
                # x 的 shape 为 [n, 1, 28, 28]，不能够直接放入到 MinstNet 进行训练，要对其结构进行转变
                # [n, 1, 28, 28] => [n,  784]
                x = x.view(x.size(0), 28 * 28)
                # 将 x 放入到模型中进行前向传播，得到的结果
                # x:[n, 784] => out:[n, 10]
                out = self.forward(x)
                #
                y_onehot = one_hot(y)
                loss = functional.mse_loss(y_onehot, out)
                train_loss.append(loss)

                optimizer.zero_grad()
                loss.backward()
                # w' = w - learning_rate * gradient
                optimizer.step()

                if iter_idx % 10 == 0:
                    print(epoch, iter_idx, loss.item())

                plot_curve(train_loss)

    def test_runner(self):
        total_correct = 0
        for x, y in self.test_loader:
            x = x.view(x.size(0), 28 * 28)
            out = self.forward(x)
            total_correct += out.argmax(dim=1).eq(y).sum().float().item()
        acc = total_correct / len(self.test_loader.dataset)
        print("test accuracy: " + str(acc))


if __name__ == "__main__":
    net = MinstNet()
    net.train_runner()
    net.test_runner()
