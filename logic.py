import torch as t
from torch.autograd import Variable as V
from matplotlib import pyplot as plt


t.manual_seed(100)

def get_data(batch_size = 8):
    x = t.rand(batch_size, 1) * 20
    y = x * 3 + (1 + t.randn(batch_size, 1)) * 5
    return x, y



w = V(t.rand(1,1), requires_grad=True)
b = V(t.rand(1,1), requires_grad=True)

lr = 0.001 #learning rate

for i in range(10000):
    x, y = get_data()
    x, y = V(x), V(y)

    y_pre = x.mm(w) + b.expand_as(y)
    loss = 0.5 * (y_pre - y) ** 2
    loss = loss.sum()

    loss.backward()

    w.data.sub_(lr * w.grad.data)
    b.data.sub_(lr * b.grad.data)

    w.grad.data.zero_()
    b.grad.data.zero_()


print(w.data.squeeze()[0], b.data.squeeze()[0])


x = t.arange(0,20).view(-1, 1)
y = x.mm(w.data) + b.data.expand_as(x)

plt.plot(x.numpy(), y.numpy())

x1, y1 = get_data(20)

plt.scatter(x1.numpy(), y1.numpy())
plt.show()
