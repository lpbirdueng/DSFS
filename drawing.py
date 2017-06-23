# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
theta = np.arange(0, 360, 1)
x = np.cos(theta)
y = np.sin(theta)
plt.title("drawing")
plt.plot(x, y, 'rx', label='circle')  # red x
plt.legend()
plt.show()














if __name__ == '__main__':
    pass