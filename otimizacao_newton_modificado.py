import numpy as np


def rodar_newton_modificado(
    funcao, gradiente_f, hessiana_f, x_inicial, iteracoes, alpha, epsilon
):
    x_atual = np.array(x_inicial, dtype=float)

    historico_x = [x_atual.copy()]
    historico_f = [funcao(x_atual)]

    for i in range(iteracoes):
        gradiente = gradiente_f(x_atual)
        hessiana = hessiana_f(x_atual)

        # Inverte a matriz Hessiana exata
        try:
            hessiana_invertida = np.linalg.inv(hessiana)
        except np.linalg.LinAlgError:
            hessiana_invertida = np.eye(len(x_atual))

        # Calcula a direção de descida
        direcao = -np.dot(hessiana_invertida, gradiente)

        # A GRANDE MODIFICAÇÃO: Usa o 'alpha' para controlar o tamanho do passo
        x_novo = x_atual + alpha * direcao

        historico_x.append(x_novo.copy())
        historico_f.append(funcao(x_novo))

        # Critério de parada
        if epsilon is not None and np.linalg.norm(x_novo - x_atual) < epsilon:
            break

        x_atual = x_novo

    return np.array(historico_x), np.array(historico_f)
