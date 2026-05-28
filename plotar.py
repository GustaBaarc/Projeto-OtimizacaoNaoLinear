import numpy as np
import matplotlib.pyplot as plt

def exibir_graficos(lista_resultados, funcao_alvo):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Descobrir os limites do gráfico juntando as coordenadas de todos os métodos
    x1_todos = []
    x2_todos = []
    for res in lista_resultados:
        x1_todos.extend(res["hist_x"][:, 0])
        x2_todos.extend(res["hist_x"][:, 1])

    # ZOOM AUTOMÁTICO DE 15% (Substitui a margem cravada de 1.0)
    margem_x1 = max((max(x1_todos) - min(x1_todos)) * 0.15, 0.2)
    margem_x2 = max((max(x2_todos) - min(x2_todos)) * 0.15, 0.2)

    x1_min, x1_max = min(x1_todos) - margem_x1, max(x1_todos) + margem_x1
    x2_min, x2_max = min(x2_todos) - margem_x2, max(x2_todos) + margem_x2

    X1_grid, X2_grid = np.meshgrid(
        np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100)
    )
    Z_grid = funcao_alvo([X1_grid, X2_grid])

    # Desenha o relevo de fundo
    ax1.contour(X1_grid, X2_grid, Z_grid, levels=40, cmap="viridis")

    # Cores para cada método
    cores = ["red", "blue", "green", "orange", "purple"]

    # 2. Desenha a linha de cada método no gráfico
    for i, res in enumerate(lista_resultados):
        cor = cores[i % len(cores)]
        nome = res["nome"]
        hist_x = res["hist_x"]
        hist_f = res["hist_f"]

        # Curva de Deslocamento (x1 vs x2)
        ax1.plot(hist_x[:, 0], hist_x[:, 1], marker="o", color=cor, linestyle="-", label=nome, markersize=4)

        # Marcação do ponto inicial e final
        if i == 0:
            ax1.plot(hist_x[0, 0], hist_x[0, 1], marker="s", color="black", markersize=8, label="Início")
        ax1.plot(hist_x[-1, 0], hist_x[-1, 1], marker="*", color=cor, markersize=10)

        # Curva de Convergência (Iteração vs f(x))
        ax2.plot(range(len(hist_f)), hist_f, marker="o", color=cor, linestyle="-", label=nome, markersize=4)

    ax1.set_title("Curvas de Nível e Deslocamento")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()

    ax2.set_title("Convergência da Função Objetivo")
    ax2.set_xlabel("Iterações")
    ax2.set_ylabel("f(x, y)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()