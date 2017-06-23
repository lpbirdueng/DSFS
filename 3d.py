# -*- coding: utf-8 -*-
'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from c08.gradientdescent import sum_of_squares_gradient
from c08.gradientdescent import sum_of_squares


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
theta = np.arange(0, 360, 1)
# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
# X = np.arange(-6, 6, 1)
# Y = np.arange(-6, 6, 1)
# X = np.arange(-1, 1, 0.01)
# Y = np.sqrt(1 - X**2)
# Y = np.arange(-1, 1, 0.1)
# u = np.cos(theta)
# X = np.cos(theta) * u
# Y = np.sin(theta) * u
X = np.cos(theta)
Y = np.sin(theta)
X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)
# Z1 = X**2 + Y**2
# Z1 = np.sqrt(1 - X**2 - Y**2)
# Z1 = np.sin(theta)
Z1 = np.sqrt(1 - X**2 - Y**2)


# Z2 = X*2 + Y*2
# Z = [sum_of_squares([x_i, y_i]) for x_i, y_i in zip(X, Y)]
# Z = X**2 + 2*X*Y + Y**2



# Plot the surface.
#surf = ax.plot_surface(X, Y, Z1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# ax.plot_surface(X, Y, Z2, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.plot_wireframe(X, Y, Z1, rstride=10, cstride=10)
# Customize the z axis.
ax.set_zlim(0, 1)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

