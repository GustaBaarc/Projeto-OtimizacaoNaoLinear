import numpy as np

def rodar_gradiente(funcao, gradiente_f, x_inicial, iteracoes, alpha, epsilon):
    x_atual = np.array(x_inicial, dtype=float)
    
    historico_x = [x_atual.copy()]
    historico_f = [funcao(x_atual)]
    
    for i in range(iteracoes):
        gradiente = gradiente_f(x_atual)
        
        # Passo do Gradiente
        x_novo = x_atual - alpha * gradiente
        
        historico_x.append(x_novo.copy())
        historico_f.append(funcao(x_novo))
        
        # Critério de parada
        if epsilon is not None and np.linalg.norm(x_novo - x_atual) < epsilon:
            break
            
        x_atual = x_novo
        
    return np.array(historico_x), np.array(historico_f)