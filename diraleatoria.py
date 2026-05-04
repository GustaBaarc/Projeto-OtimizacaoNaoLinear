import numpy as np

def rodar_aleatoria(funcao, x_inicial, iteracoes, direcoes_lista, alpha, epsilon):
    x_atual = np.array(x_inicial, dtype=float)
    
    historico_x = [x_atual.copy()]
    historico_f = [funcao(x_atual)]
    
    for i in range(iteracoes):
        if direcoes_lista is not None and i < len(direcoes_lista):
            direcao = np.array(direcoes_lista[i])
        else:
            direcao = np.random.randn(2)
            direcao = direcao / np.linalg.norm(direcao)
            
        x_novo = x_atual + alpha * direcao
        
        if funcao(x_novo) < funcao(x_atual):
            x_atual = x_novo
            
        historico_x.append(x_atual.copy())
        historico_f.append(funcao(x_atual))
        
        if epsilon is not None and i > 0:
            erro = abs(historico_f[-1] - historico_f[-2])
            if 0 < erro < epsilon:
                break
                
    return np.array(historico_x), np.array(historico_f)