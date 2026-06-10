import numpy as np
from criterios_parada import verificar_parada


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


def rodar_newton_modificado(
    funcao,
    gradiente_f,
    hessiana_f,
    x_inicial,
    iteracoes,
    epsilon,
    criterio="delta_f",  # "delta_f" | "norma_x" | "norma_grad" | "max_iter"
    max_iter_fixo=None,
):
    """
    Método de Newton Modificado (com busca de linha pela seção áurea).

    Critérios de parada disponíveis (parâmetro `criterio`):
        "delta_f"    – variação relativa de f nos últimos 6 passos  (padrão original)
        "norma_x"    – ‖x_novo − x_ant‖ < epsilon
        "norma_grad" – ‖∇f(x)‖ < epsilon
        "max_iter"   – para após `max_iter_fixo` iterações
    """
    x_at = np.array(x_inicial, dtype=float)
    hx = [x_at.copy()]
    hf = [funcao(x_at)]

    for it in range(iteracoes):
        g = gradiente_f(x_at)
        H = hessiana_f(x_at)

        try:
            H_inv = np.linalg.inv(H)
        except Exception:
            H_inv = np.eye(len(x_at))

        direcao = -np.dot(H_inv, g)
        alpha = busca_secao_aurea(lambda a: funcao(x_at + a * direcao))
        x_nv = x_at + alpha * direcao

        hx.append(x_nv.copy())
        hf.append(funcao(x_nv))

        if verificar_parada(
            criterio,
            epsilon,
            hf,
            x_novo=x_nv,
            x_ant=x_at,
            gradiente=gradiente_f(x_nv),
            iteracao_atual=it,
            max_iter_fixo=max_iter_fixo,
        ):
            x_at = x_nv
            break

        x_at = x_nv

    return np.array(hx), np.array(hf)
