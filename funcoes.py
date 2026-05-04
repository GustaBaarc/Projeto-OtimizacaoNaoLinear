import numpy as np


def f_ex1(x):
    """Função do Exercício 1"""
    x1, x2 = x[0], x[1]
    return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2


def f_three_hump_camel(x):
    """Função Three-Hump Camel (Exercícios 4 e 5)"""
    x1, x2 = x[0], x[1]
    return 2 * (x1**2) - 1.05 * (x1**4) + (x1**6) / 6 + x1 * x2 + (x2**2)
