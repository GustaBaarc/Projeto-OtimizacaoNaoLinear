import numpy as np
from scipy.optimize import minimize


def busca_aleatoria(f, x0, max_iter, direcoes=None, alpha=0.1):
    x = np.array(x0, dtype=float)
    hist_x = [x.copy()]
    hist_f = [f(x)]

    for k in range(max_iter):
        # Verifica se o usuário mandou uma lista de direções no main.py
        if direcoes is not None and k < len(direcoes):
            d = np.array(direcoes[k])
        else:
            # Caso contrário, gera uma direção 100% aleatória e normaliza o vetor
            d = np.random.randn(2)
            d = d / np.linalg.norm(d)

        x_novo = x + alpha * d

        # Só anda para o novo ponto se houver minimização da função
        if f(x_novo) < f(x):
            x = x_novo

        hist_x.append(x.copy())
        hist_f.append(f(x))

    return np.array(hist_x), np.array(hist_f)


def otimizador_scipy(f, x0, metodo_scipy="BFGS"):
    hist_x = [np.array(x0)]
    hist_f = [f(x0)]

    def registrador_de_passos(xk):
        hist_x.append(xk.copy())
        hist_f.append(f(xk))

    minimize(f, x0, method=metodo_scipy, callback=registrador_de_passos)
    return np.array(hist_x), np.array(hist_f)
