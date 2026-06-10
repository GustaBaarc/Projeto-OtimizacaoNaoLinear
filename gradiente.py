import numpy as np
from criterios_parada import verificar_parada


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


def rodar_gradiente(
    funcao,
    gradiente_f,
    x_inicial,
    iteracoes,
    epsilon,
    criterio="delta_f",
    max_iter_fixo=None,
):
    x_at = np.array(x_inicial, dtype=float)
    hx = [x_at.copy()]
    hf = [funcao(x_at)]
    hg = [float(np.linalg.norm(gradiente_f(x_at)))]

    for it in range(iteracoes):
        g = gradiente_f(x_at)
        direcao = -g
        alpha = busca_secao_aurea(lambda a: funcao(x_at + a * direcao))
        x_nv = x_at + alpha * direcao
        g_nv = gradiente_f(x_nv)

        hx.append(x_nv.copy())
        hf.append(funcao(x_nv))
        hg.append(float(np.linalg.norm(g_nv)))

        if verificar_parada(
            criterio, epsilon, hf,
            x_novo=x_nv, x_ant=x_at,
            gradiente=g_nv,
            iteracao_atual=it,
            max_iter_fixo=max_iter_fixo,
        ):
            x_at = x_nv
            break

        x_at = x_nv

    return np.array(hx), np.array(hf), np.array(hg)