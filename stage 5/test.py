import numpy as np
from matplotlib import pyplot as plt

plt.plot(np.arange(0, 4, 1, dtype=np.int), [3.4, 5.6, 7.8, 9.9])
plt.ylabel("score")
plt.xlabel("episode")
plt.title("Training Scores")
plt.show()