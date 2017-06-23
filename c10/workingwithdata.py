# -*- coding: utf-8 -*-
import math
from collections import Counter
import matplotlib.pyplot as plt
import random
from c06.probability import inverse_normal_cdf
from c04.Linear_Algebra import shape
from c04.Linear_Algebra import get_column
from c04.Linear_Algebra import make_matrix
from c05.statistics import correlation
import dateutil.parser
import csv


def bucketize(point, bucket_size):
    """floor the point to the next lower multiple of bucket_size"""
    return bucket_size * math.floor(point/bucket_size)


def make_histogram(points, bucket_size):
    """buckets the points and counts how many in each bucket"""
    return Counter(bucketize(point, bucket_size) for point in points)


def plot_histogram(points, bucket_size, title=""):
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
    plt.title(title)
    plt.show()


def random_normal():
    """returns a random draw from a standard normal distribution"""
    return inverse_normal_cdf(random.random())


def parse_row(input_row, parsers):
    """given a list of parsers (some of which may be None)
    apply the appropriate one to each element of the input_row"""
    return [try_or_none(parser)(value) if parser is not None else value
            for value, parser in zip(input_row, parsers)]


def parse_rows_with(reader, parsers):
    """wrap a reader to apply the parsers to each of its rows"""
    for row in reader:
        yield parse_row(row, parsers)


def try_or_none(f):
    """wraps f to return None if f raises an exception
    assumes f takes only one input"""
    def f_or_none(x):
        try: return f(x)
        except: return None
    return f_or_none


def correlation_matrix(data):
    """returns the num_columns x num_columns matrix whose(i,j)th entry
    is the correlation between columns i and j of data"""
    _, num_columns = shape(data)

    def matrix_entry(i, j):
        return correlation(get_column(data, i), get_column(data, j))
    return make_matrix(num_columns, num_columns, matrix_entry)


def try_parse_field(field_name, value, parser_dict):
    """try to parse value using the appropriate function from parser_dict"""
    parser = parser_dict.get(field_name)        # None if no such entry
    if parser is not None:
        return try_or_none(parser)(value)
    else:
        return value


def parse_dict(input_dict, parser_dict):
    return { field_name : try_parse_field(field_name, value, parser_dict)
             for field_name, value in input_dict.iteritems()}


if __name__ == '__main__':
    data = []
    with open("comma_delimited_stock_prices.csv", newline='') as f:
        reader = csv.reader(f)
        for line in parse_rows_with(reader, [dateutil.parser.parser, None, float]):
            data.append(line)
            # print(line)
        print(data)
        for row in data:
            if any(x is None for x in row):
                print(row[0].info)

    """
    xs = [random_normal() for _ in range(1000)]
    ys1 = [x + random_normal() / 2 for x in xs]
    ys2 = [-x + random_normal() / 2 for x in xs]

    plt.scatter(xs, ys1, marker='.', c='black', label='ys1')
    plt.scatter(xs, ys2, marker='.', c='grey', label='ys2')
    plt.xlabel('xs')
    plt.ylabel('ys')
    plt.legend()
    plt.show()
    """
    """
    random.seed(0)

    # uniform between -100 and 100
    uniform = [200 * random.random() - 100 for _ in range(10000)]

    # normal distribution with mean 0, standard deviation 57
    normal = [57 * inverse_normal_cdf(random.random())
              for _ in range(10000)]

    plot_histogram(uniform, 10, "Uniform Histogram")

    plot_histogram(normal, 10, "Normal Histogram")
    """

    pass