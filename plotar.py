import numpy as np
import matplotlib.pyplot as plt


def exibir_graficos(hist_x, hist_f, f_alvo):
    print(">> Gerando gráficos... Feche a janela para encerrar o script.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Gráfico 1: Curvas de Nível e Deslocamento ---
    margem = 1.0
    x1_min, x1_max = min(hist_x[:, 0]) - margem, max(hist_x[:, 0]) + margem
    x2_min, x2_max = min(hist_x[:, 1]) - margem, max(hist_x[:, 1]) + margem

    X1_grid, X2_grid = np.meshgrid(
        np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100)
    )

    # O NumPy consegue calcular a matriz inteira de uma vez só
    Z_grid = f_alvo([X1_grid, X2_grid])

    ax1.contour(X1_grid, X2_grid, Z_grid, levels=30, cmap="viridis")
    ax1.plot(hist_x[:, 0], hist_x[:, 1], marker="o", color="red", linestyle="-")
    ax1.plot(hist_x[0, 0], hist_x[0, 1], marker="s", color="black", label="Início")

    ax1.set_title("Curva de Deslocamento")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.legend()

    # --- Gráfico 2: Curva de Convergência ---
    ax2.plot(range(len(hist_f)), hist_f, marker="o", color="blue")
    ax2.set_title("Curva de Convergência")
    ax2.set_xlabel("Iterações")
    ax2.set_ylabel("F(x1, x2)")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()
