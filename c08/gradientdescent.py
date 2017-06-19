# -*- coding: utf-8 -*-
from functools import partial
import matplotlib.pyplot as plt
import random
from c04.Linear_Algebra import distance


def sum_of_squares(v):
    """computes the sum of squared elements in v"""
    return sum(v_i ** 2 for v_i in v)


def difference_quotient(f, x, h):
    return (f(x + h) - f(x)) / h


def square(x):
    return x * x


def derivation(x):
    return 2 * x


def partial_difference_quotient(f, v, i, h):
    """compute the ith partial difference quotient of f at v"""
    w = [v_j + (h if j == i else 0)         # add h to just the ith element of v
         for j, v_j in enumerate(v)]
    return (f(w) -  f(v)) / h


def estimate_gradient(f, v, h = 0.00001):
    return [partial_difference_quotient(f, v, i, h)
            for i, _ in enumerate(v)]


def step(v, direction, step_size):
    """move step_size in the direction from v"""
    return [v_i + step_size * direction_i
            for v_i, direction_i in zip(v, direction)]


def sum_of_squares_gradient(v):
    return [2 * v_i for v_i in v]


if __name__ == '__main__':
    v = [random.randint(-10, 10) for i in range(3)]
    tolerance = 0.000000000000001

    while True:
        gradient = sum_of_squares_gradient(v)   # compute the gradient at v
        next_v = step(v, gradient, -0.0000001)       # take a negative gradient step
        if distance(next_v, v) < tolerance:     # stop if we're converging
            print(next_v)
            break
        v = next_v                              # continue if we're not
    exit()
    derivative_estimate = partial(difference_quotient, square, h=0.01)
    x = range(-10, 10)
    plt.title("Actual Derivatives vs. Estimates")
    plt.plot(x, list(map(derivation, x)), 'rx', label='Actual')           # red x
    plt.plot(x, list(map(derivative_estimate, x)), 'b+', label='Estimate')    # blue +
    plt.legend(loc=9)
    plt.show()

    # plot to show they're basically the same
    pass