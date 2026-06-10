import numpy as np
from criterios_parada import verificar_parada


def rodar_newton(
    funcao,
    gradiente_f,
    hessiana_f,
    x_inicial,
    iteracoes,
    epsilon,
    criterio="delta_f",     # "delta_f" | "norma_x" | "norma_grad" | "max_iter"
    max_iter_fixo=None,
):
    """
    Método de Newton puro (passo completo, sem busca de linha).

    Critérios de parada disponíveis (parâmetro `criterio`):
        "delta_f"    – variação relativa de f nos últimos 6 passos  (padrão original)
        "norma_x"    – ‖x_novo − x_ant‖ < epsilon
        "norma_grad" – ‖∇f(x)‖ < epsilon
        "max_iter"   – para após `max_iter_fixo` iterações
                       (Questão 2: Dixon-Price com Newton em 1 iteração)
    """
    x_at = np.array(x_inicial, dtype=float)
    hx = [x_at.copy()]
    hf = [funcao(x_at)]

    for it in range(iteracoes):
        g = gradiente_f(x_at)
        H = hessiana_f(x_at)

        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            H_inv = np.eye(len(x_at))

        x_nv = x_at - np.dot(H_inv, g)

        hx.append(x_nv.copy())
        hf.append(funcao(x_nv))

        if verificar_parada(
            criterio, epsilon, hf,
            x_novo=x_nv, x_ant=x_at,
            gradiente=gradiente_f(x_nv),
            iteracao_atual=it,
            max_iter_fixo=max_iter_fixo,
        ):
            x_at = x_nv
            break

        x_at = x_nv

    return np.array(hx), np.array(hf)