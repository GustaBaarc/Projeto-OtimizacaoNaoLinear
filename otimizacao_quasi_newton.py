import numpy as np

def rodar_quasi_newton(funcao, gradiente_f, x_inicial, iteracoes, alfas_lista, alpha_padrao, epsilon):
    x_atual = np.array(x_inicial, dtype=float)
    
    historico_x = [x_atual.copy()]
    historico_f = [funcao(x_atual)]
    
    # Inicia com Matriz Identidade
    H = np.eye(len(x_atual))
    
    for i in range(iteracoes):
        gradiente = gradiente_f(x_atual)
        
        # Pega o alpha da lista (se existir) ou usa o padrão
        if alfas_lista is not None and i < len(alfas_lista):
            alpha = alfas_lista[i]
        else:
            alpha = alpha_padrao
            
        direcao = -np.dot(H, gradiente)
        x_novo = x_atual + alpha * direcao
        
        # Atualização BFGS
        s = x_novo - x_atual
        y = gradiente_f(x_novo) - gradiente
        sy = np.dot(s, y)
        
        if sy > 1e-10:
            rho = 1.0 / sy
            I = np.eye(len(x_atual))
            H = np.dot(np.dot((I - rho * np.outer(s, y)), H), (I - rho * np.outer(y, s))) + rho * np.outer(s, s)
            
        historico_x.append(x_novo.copy())
        historico_f.append(funcao(x_novo))
        
        if epsilon is not None and np.linalg.norm(x_novo - x_atual) < epsilon:
            break
            
        x_atual = x_novo
        
    return np.array(historico_x), np.array(historico_f)