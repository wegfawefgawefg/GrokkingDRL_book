import torch

mus = torch.tensor([0.0, 4.0, 10.0])
sigmas = torch.tensor([0.01, 0.01, 0.01])
distributions = torch.distributions.Normal(mus, sigmas)
# samples = distributions.sample(sample_shape=torch.Size([3]))
samples = distributions.sample()
print(samples)
q = distributions.log_prob(samples)
print(q)
