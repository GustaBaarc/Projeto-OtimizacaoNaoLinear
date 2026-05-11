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


def rodar_aleatoria(funcao, x_inicial, iteracoes, epsilon):
    x_at = np.array(x_inicial, dtype=float)
    hx, hf = [x_at.copy()], [funcao(x_at)]
    for _ in range(iteracoes):
        d = np.random.randn(2)
        d /= np.linalg.norm(d)
        alpha = busca_secao_aurea(lambda a: funcao(x_at + a * d))
        x_at += alpha * d
        hx.append(x_at.copy())
        hf.append(funcao(x_at))
        if len(hf) >= 6:
            df_tot = max(hf) - min(hf)
            if df_tot > 1e-10 and (max(hf[-6:]) - min(hf[-6:])) < epsilon * df_tot:
                break
    return np.array(hx), np.array(hf)
