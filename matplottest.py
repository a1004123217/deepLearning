import matplotlib.pyplot as plt

import numpy as np

x = np.linspace(-2, 2, 100)

y = 3 * x + 4.0
y2 = x ** 2

plt.plot(x, y)
plt.plot(x ,y2)
plt.show()