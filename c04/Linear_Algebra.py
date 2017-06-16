# -*- coding: utf-8 -*-
from functools import reduce
import math


def vector_add(v, w):
    """ adds corresponding elements"""
    return [v_i + w_i
            for v_i, w_i in zip(v, w)]


def vector_substract(v, w):
    '''substracts corresponding elements'''
    return [v_i - w_i
            for v_i, w_i in zip(v, w)]


def vector_sum(vectors):
    '''sums all correcponding elements'''
    return reduce(vector_add, vectors)
    """
    result = vectors[0]
    for vector in vectors[1:]:
        result = vector_add(result, vector)
    return result
    """


def scalar_multiply(c, v):
    '''c is a number, v is a vector'''
    return [c * v_i for v_i in v]


def vecotr_mean(vectors):
    '''compute the vector whose ith element is the mean of the ith elements of the input vectors'''
    n = len(vectors)
    return scalar_multiply(1 / n, vector_sum(vectors))


def dot(v, w):
    '''v_1 * w_1 + ...+v_n * w_n'''
    return sum(v_i * w_i
               for v_i, w_i in zip(v, w))


def sum_of_squares(v):
    '''v_1 * v_1 + ...+ v_n * v_n'''
    return dot(v, v)


def magnitude(v):
    return math.sqrt(sum_of_squares())


def squared_distance(v, w):
    '''(v_1 - w_1)**2 + ... + (v_n - w_n)**2'''
    return sum_of_squares(vector_substract(v, w))


def distance(v, w):
    return math.sqrt(squared_distance(v, w))


""" matrix"""


def shape(A):
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0  # number of elements in first row
    return num_rows, num_cols


def get_row(A, i):
    return A[i]


def get_column(A, j):
    return [A_i[j]
            for A_i in A]


def make_matrix(num_rows, num_cols, entry_fn):
    '''returns a num_rows x num_cols matrix
    whose (i,j)th entry is entry_fn(i, j)'''
    return [[entry_fn(i, j)  # given i, create a list
             for j in range(num_cols)]  # [entry_fn(i, 0), ...]
            for i in range(num_rows)]  # create one list for each i


def is_diagonal(i, j):
    '''1's on the 'diagonal', 0's everywhere else'''
    return 1 if i == j else 0


identity_matrix = make_matrix(5, 8, is_diagonal)
print(identity_matrix)
