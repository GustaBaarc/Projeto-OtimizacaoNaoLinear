import numpy as np


def busca_secao_aurea(f_alpha, a=0.0, b=2.0, tol=1e-4):
    r = 0.618
    x1, x2 = b - r * (b - a), a + r * (b - a)
    f1, f2 = f_alpha(x1), f_alpha(x2)
    while (b - a) > tol:
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - r * (b - a)
            f1 = f_alpha(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + r * (b - a)
            f2 = f_alpha(x2)
    return (a + b) / 2.0


def rodar_gradiente(funcao, gradiente_f, x_inicial, iteracoes, epsilon):
    x_at = np.array(x_inicial, dtype=float)
    hx, hf = [x_at.copy()], [funcao(x_at)]
    for _ in range(iteracoes):
        direcao = -gradiente_f(x_at)
        alpha = busca_secao_aurea(lambda a: funcao(x_at + a * direcao))
        x_at = x_at + alpha * direcao
        hx.append(x_at.copy())
        hf.append(funcao(x_at))
        if len(hf) >= 6:
            df_tot = max(hf) - min(hf)
            df_5 = max(hf[-6:]) - min(hf[-6:])
            if df_tot > 1e-10 and df_5 < epsilon * df_tot:
                break
    return np.array(hx), np.array(hf)
