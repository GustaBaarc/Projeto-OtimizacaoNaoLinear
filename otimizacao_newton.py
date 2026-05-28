import numpy as np


def rodar_newton(funcao, gradiente_f, hessiana_f, x_inicial, iteracoes, epsilon):
    x_atual = np.array(x_inicial, dtype=float)
    historico_x = [x_atual.copy()]
    historico_f = [funcao(x_atual)]

    for _ in range(iteracoes):
        gradiente = gradiente_f(x_atual)
        hessiana = hessiana_f(x_atual)

        try:
            hessiana_invertida = np.linalg.inv(hessiana)
        except np.linalg.LinAlgError:
            hessiana_invertida = np.eye(len(x_atual))

        passo = np.dot(hessiana_invertida, gradiente)
        x_novo = x_atual - passo

        historico_x.append(x_novo.copy())
        historico_f.append(funcao(x_novo))

        if len(historico_f) >= 6:
            delta_f_total = max(historico_f) - min(historico_f)
            ultimos_6 = historico_f[-6:]
            delta_f_5 = max(ultimos_6) - min(ultimos_6)
            if delta_f_total > 1e-10 and delta_f_5 < epsilon * delta_f_total:
                break

        x_atual = x_novo

    return np.array(historico_x), np.array(historico_f)
