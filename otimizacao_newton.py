import numpy as np

def rodar_newton(funcao, gradiente_f, hessiana_f, x_inicial, iteracoes, epsilon):
    x_atual = np.array(x_inicial, dtype=float)
    
    historico_x = [x_atual.copy()]
    historico_f = [funcao(x_atual)]
    
    for i in range(iteracoes):
        gradiente = gradiente_f(x_atual)
        hessiana = hessiana_f(x_atual)
        
        # Inverte a matriz e calcula o passo
        hessiana_invertida = np.linalg.inv(hessiana)
        passo = np.dot(hessiana_invertida, gradiente)
        
        x_novo = x_atual - passo
        
        historico_x.append(x_novo.copy())
        historico_f.append(funcao(x_novo))
        
        if epsilon is not None and np.linalg.norm(x_novo - x_atual) < epsilon:
            break
            
        x_atual = x_novo
        
    return np.array(historico_x), np.array(historico_f)