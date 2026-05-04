from funcoes import f_ex1, f_three_hump_camel
from otimizadores import busca_aleatoria, otimizador_scipy
from plotar import exibir_graficos


def main():

    f_alvo = f_ex1
    x0 = [0.5, 3.5]
    iteracoes = 5
    passo_alpha = 0.1

    # Vetor de Direções
    direcoes = [
        [-0.3850, 0.6680],
        [0.4560, -0.4956],
        [0.8570, 0.1049],
        [0.4956, -0.3456],
        [0.6680, 0.7829],
    ]
    # direcoes = None

    # ==========================================
    # EXECUÇÃO DO MOTOR
    # ==========================================
    print("\n" + "-" * 60)
    print(" INICIANDO OTIMIZAÇÃO (BUSCA ALEATÓRIA)")
    print("-" * 60)

    hist_x, hist_f = busca_aleatoria(
        f_alvo, x0, max_iter=iteracoes, direcoes=direcoes, alpha=passo_alpha
    )

    # --- NOVO BLOCO: Imprimindo cada passo ---
    print("\nHISTÓRICO PASSO A PASSO:")
    for i in range(len(hist_f)):
        # i=0 é o ponto inicial. A partir de i=1 são as iterações.
        nome_passo = "Início (X0)" if i == 0 else f"Iteração {i:02d}"
        print(
            f"{nome_passo}: [x1: {hist_x[i][0]:.6f}, x2: {hist_x[i][1]:.6f}]  ->  F(x): {hist_f[i]:.6f}"
        )

    print("\n" + "-" * 60)
    print(" RESUMO FINAL")
    print("-" * 60)
    print(f"Total de passos executados: {len(hist_f) - 1}")
    print(
        f"Ponto de Mínimo Encontrado [x1, x2]: [{hist_x[-1][0]:.6f}, {hist_x[-1][1]:.6f}]"
    )
    print(f"Valor Final da Função F(x): {hist_f[-1]:.6f}")
    print("-" * 60 + "\n")

    # Exibe os gráficos
    exibir_graficos(hist_x, hist_f, f_alvo)


if __name__ == "__main__":
    main()
