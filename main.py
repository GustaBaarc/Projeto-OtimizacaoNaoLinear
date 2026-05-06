from funcoes import f1, grad1, hess1, f4, grad4, hess4
from gradiente import rodar_gradiente
from otimizacao_newton import rodar_newton
from otimizacao_quasi_newton import rodar_quasi_newton
from diraleatoria import rodar_aleatoria
from plotar import exibir_graficos
from otimizacao_newton_modificado import rodar_newton_modificado


def main():
    # Digite aqui os métodos que deseja rodar dentro de uma lista.
    METODOS_ESCOLHIDOS = [5]

    # Variáveis gerais
    ponto_inicial = [0.5, 0.5]
    iteracoes_maximas = 50
    tamanho_passo = 0.05
    criterio_parada = 0.001

    # Alfas para Metodo 3 (Quasi-Newton)
    alfas_lista = [0.4125, 0.7530]
    resultados_para_plotar = []
    funcao_fundo = f1

    # ==============================================================================
    # 1. MÉTODO DO VETOR GRADIENTE
    # ==============================================================================
    if 1 in METODOS_ESCOLHIDOS:
        print("\nRodando: VETOR GRADIENTE...")
        funcao_alvo = f1
        funcao_fundo = f1

        hist_x, hist_f = rodar_gradiente(
            funcao=funcao_alvo,
            gradiente_f=grad1,
            x_inicial=ponto_inicial,
            iteracoes=iteracoes_maximas,
            alpha=tamanho_passo,
            epsilon=criterio_parada,
        )
        resultados_para_plotar.append(
            {"nome": "Gradiente", "hist_x": hist_x, "hist_f": hist_f}
        )

    # ==============================================================================
    # 2. MÉTODO DE NEWTON
    # ==============================================================================
    if 2 in METODOS_ESCOLHIDOS:
        print("\nRodando: NEWTON...")
        funcao_alvo = f1
        funcao_fundo = f1

        hist_x, hist_f = rodar_newton(
            funcao=funcao_alvo,
            gradiente_f=grad1,
            hessiana_f=hess1,
            x_inicial=ponto_inicial,
            iteracoes=iteracoes_maximas,
            epsilon=criterio_parada,
        )
        resultados_para_plotar.append(
            {"nome": "Newton", "hist_x": hist_x, "hist_f": hist_f}
        )

    # ==============================================================================
    # 3. MÉTODO QUASI-NEWTON (BFGS)
    # ==============================================================================
    if 3 in METODOS_ESCOLHIDOS:
        print("\nRodando: QUASI-NEWTON (BFGS)...")
        funcao_alvo = f4
        funcao_fundo = (
            f4  # Atualiza o fundo porque o Ex 4 e 5 usa a função Three-Hump Camel
        )
        ponto_inicial_qnewton = [0.5, 0.5]

        hist_x, hist_f = rodar_quasi_newton(
            funcao=funcao_alvo,
            gradiente_f=grad4,
            x_inicial=ponto_inicial_qnewton,
            iteracoes=iteracoes_maximas,
            alfas_lista=alfas_lista,
            alpha_padrao=tamanho_passo,
            epsilon=criterio_parada,
        )
        resultados_para_plotar.append(
            {"nome": "Quasi-Newton", "hist_x": hist_x, "hist_f": hist_f}
        )

    # ==============================================================================
    # 4. BUSCA ALEATÓRIA
    # ==============================================================================
    if 4 in METODOS_ESCOLHIDOS:
        print("\nRodando: BUSCA ALEATÓRIA...")
        funcao_alvo = f1
        funcao_fundo = f1

        hist_x, hist_f = rodar_aleatoria(
            funcao=funcao_alvo,
            x_inicial=ponto_inicial,
            iteracoes=iteracoes_maximas,
            direcoes_lista=None,
            alpha=0.1,
            epsilon=criterio_parada,
        )
        resultados_para_plotar.append(
            {"nome": "Busca Aleatória", "hist_x": hist_x, "hist_f": hist_f}
        )

    # ==============================================================================
    # 5. MÉTODO DE NEWTON MODIFICADO (Questão 4)
    # ==============================================================================
    if 5 in METODOS_ESCOLHIDOS:
        print("\nRodando: NEWTON MODIFICADO...")

        # Configure aqui a função que a Questão 4 pede (Ex: f4 e hess4)
        funcao_alvo = f4
        funcao_fundo = f4

        hist_x, hist_f = rodar_newton_modificado(
            funcao=funcao_alvo,
            gradiente_f=grad4,
            hessiana_f=hess4,
            x_inicial=ponto_inicial,
            iteracoes=iteracoes_maximas,
            alpha=tamanho_passo,
            epsilon=criterio_parada,
        )
        resultados_para_plotar.append(
            {"nome": "Newton Modificado", "hist_x": hist_x, "hist_f": hist_f}
        )

    # ==============================================================================
    # RESULTADOS NA TELA
    # ==============================================================================
    print("\n" + "=" * 40)
    print("HISTÓRICO DE PASSOS POR MÉTODO")
    print("=" * 40)

    # Como agora temos vários métodos, ele faz um loop para imprimir cada um
    for res in resultados_para_plotar:
        print(f"\n--- {res['nome'].upper()} ---")
        for i in range(len(res["hist_f"])):
            print(
                f"Iteração {i}: x=[{res['hist_x'][i][0]:.5f}, {res['hist_x'][i][1]:.5f}] | f(x)={res['hist_f'][i]:.5f}"
            )

    # Só lembre de garantir que seu plotar.py é aquele atualizado que aceita receber a lista inteira!
    exibir_graficos(resultados_para_plotar, funcao_fundo)


if __name__ == "__main__":
    main()
