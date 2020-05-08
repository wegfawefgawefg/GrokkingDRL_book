import torch
from matplotlib import pyplot as plt

a = torch.tensor([0.0, 1.0], requires_grad = True)
norm = torch.distributions.normal.Normal(*a)
numSamples = 1000
xs = torch.arange(0, numSamples)
print(xs)
ys = norm.sample((numSamples, ))
ys = sorted(ys.tolist())
print(ys)
plt.plot(xs, ys)
plt.show()

print(a)

target = 4
alpha = 0.01
for i in range(300):
    norm = torch.distributions.normal.Normal(*a)
    sample = norm.rsample()
    err = (target - sample).pow(2)
    err.backward()
    print(a.grad)
    with torch.no_grad():
        a -= alpha * a.grad
        a.grad.zero_()

print(a)

norm = torch.distributions.normal.Normal(*a)
numSamples = 1000
xs = torch.arange(0, numSamples)
print(xs)
ys = norm.sample((numSamples, ))
ys = sorted(ys.tolist())
print(ys)
plt.plot(xs, ys)
plt.show()