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


def rodar_quasi_newton(
    funcao,
    gradiente_f,
    x_inicial,
    iteracoes,
    epsilon,
    criterio="delta_f",
    max_iter_fixo=None,
):
    """
    Método Quasi-Newton (BFGS) com busca de linha pela seção áurea.

    Critérios de parada disponíveis (parâmetro `criterio`):
        "delta_f"    – variação relativa de f nos últimos 6 passos
        "norma_x"    – ‖x_novo − x_ant‖ < epsilon  (padrão deste módulo)
        "norma_grad" – ‖∇f(x)‖ < epsilon
        "max_iter"   – para após `max_iter_fixo` iterações
    """
    x_at = np.array(x_inicial, dtype=float)
    H = np.eye(len(x_at))
    hx  = [x_at.copy()]
    hf  = [funcao(x_at)]
    hdx = [0.0]  # norma do passo (0 na iteração inicial)

    for it in range(iteracoes):
        g = gradiente_f(x_at)
        d = -np.dot(H, g)

        alpha = busca_secao_aurea(lambda a: funcao(x_at + a * d))
        x_nv = x_at + alpha * d

        s = x_nv - x_at
        y = gradiente_f(x_nv) - g
        sy = np.dot(s, y)

        if sy > 1e-10:
            rho = 1.0 / sy
            I = np.eye(len(x_at))
            H = (
                np.dot(
                    np.dot((I - rho * np.outer(s, y)), H),
                    (I - rho * np.outer(y, s)),
                )
                + rho * np.outer(s, s)
            )

        hx.append(x_nv.copy())
        hf.append(funcao(x_nv))
        hdx.append(float(np.linalg.norm(x_nv - x_at)))

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

    return np.array(hx), np.array(hf), np.array(hdx)