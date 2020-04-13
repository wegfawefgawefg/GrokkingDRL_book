import numpy as np
import matplotlib.pyplot as plt



mu, sigma = 0, 0.1 # mean and standard deviation
mu = np.array([0, 0.5])
sigma = np.array([0.1]*2)
s = np.random.normal(mu, sigma)
count, bins, ignored = plt.hist(s, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
plt.show()