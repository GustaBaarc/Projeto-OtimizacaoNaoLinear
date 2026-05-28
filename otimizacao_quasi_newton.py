import numpy as np


def busca_secao_aurea(f_alpha, a=0.0, b=2.0, tol=1e-4):
    r = 0.618
    x1, x2 = b - r * (b - a), a + r * (b - a)
    while (b - a) > tol:
        if f_alpha(x1) < f_alpha(x2):
            b, x2 = x2, x1
            x1 = b - r * (b - a)
        else:
            a, x1 = x1, x2
            x2 = a + r * (b - a)
    return (a + b) / 2.0


def rodar_quasi_newton(funcao, gradiente_f, x_inicial, iteracoes, epsilon):
    x_at = np.array(x_inicial, dtype=float)
    hx, hf, H = [x_at.copy()], [funcao(x_at)], np.eye(len(x_at))
    for _ in range(iteracoes):
        g = gradiente_f(x_at)
        d = -np.dot(H, g)
        alpha = busca_secao_aurea(lambda a: funcao(x_at + a * d))
        x_nv = x_at + alpha * d
        s, y = x_nv - x_at, gradiente_f(x_nv) - g
        sy = np.dot(s, y)
        if sy > 1e-10:
            rho = 1.0 / sy
            I = np.eye(len(x_at))
            H = np.dot(
                np.dot((I - rho * np.outer(s, y)), H), (I - rho * np.outer(y, s))
            ) + rho * np.outer(s, s)
        x_at = x_nv
        hx.append(x_at.copy())
        hf.append(funcao(x_at))
        if len(hf) >= 6:
            df_tot = max(hf) - min(hf)
            if df_tot > 1e-10 and (max(hf[-6:]) - min(hf[-6:])) < epsilon * df_tot:
                break
    return np.array(hx), np.array(hf)
