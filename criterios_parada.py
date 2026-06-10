import numpy as np


def verificar_parada(
    criterio: str,
    epsilon: float,
    hist_f: list,
    x_novo=None,
    x_ant=None,
    gradiente=None,
    iteracao_atual: int = 0,
    max_iter_fixo: int = None,
) -> bool:
    """
    Retorna True se o critério de parada foi atingido.

    Parâmetros
    ----------
    criterio       : "delta_f" | "norma_x" | "norma_grad" | "max_iter"
    epsilon        : tolerância / número de iterações (para max_iter)
    hist_f         : lista de valores f já calculados (incluindo o atual)
    x_novo         : ponto atual  (array)
    x_ant          : ponto anterior (array)
    gradiente      : ∇f(x_novo)  (array, opcional)
    iteracao_atual : índice 0-based da iteração corrente
    max_iter_fixo  : usado só com criterio="max_iter"; número de iterações desejadas
    """

    if criterio == "delta_f":
        # ── Variação relativa do valor da função
        if len(hist_f) < 6:
            return False
        delta_total = max(hist_f) - min(hist_f)
        delta_recente = max(hist_f[-6:]) - min(hist_f[-6:])
        return delta_total > 1e-10 and delta_recente < epsilon * delta_total

    elif criterio == "norma_x":
        # ── Convergência das variáveis 
        if x_novo is None or x_ant is None:
            return False
        return float(np.linalg.norm(x_novo - x_ant)) < epsilon

    elif criterio == "norma_grad":
        # ── Norma do gradiente 
        if gradiente is None:
            return False
        return float(np.linalg.norm(gradiente)) < epsilon

    elif criterio == "max_iter":
        # ── Número fixo de iterações
        limite = max_iter_fixo if max_iter_fixo is not None else int(epsilon)
        return iteracao_atual >= limite - 1 

    else:
        raise ValueError(
            f"Critério '{criterio}' desconhecido. "
            "Use: 'delta_f', 'norma_x', 'norma_grad' ou 'max_iter'."
        )