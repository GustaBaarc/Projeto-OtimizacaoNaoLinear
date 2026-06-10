import numpy as np
import matplotlib.pyplot as plt


def exibir_graficos(lista_resultados, funcao_alvo):
    tem_grad = any("hist_g" in res for res in lista_resultados)
    tem_dx   = any("hist_dx" in res for res in lista_resultados)
    tem_extra = tem_grad or tem_dx
    ncols = 3 if tem_extra else 2

    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    ax1, ax2 = axes[0], axes[1]
    ax3 = axes[2] if tem_extra else None

    x1_todos, x2_todos = [], []
    for res in lista_resultados:
        x1_todos.extend(res["hist_x"][:, 0])
        x2_todos.extend(res["hist_x"][:, 1])

    margem_x1 = max((max(x1_todos) - min(x1_todos)) * 0.15, 0.2)
    margem_x2 = max((max(x2_todos) - min(x2_todos)) * 0.15, 0.2)

    x1_min, x1_max = min(x1_todos) - margem_x1, max(x1_todos) + margem_x1
    x2_min, x2_max = min(x2_todos) - margem_x2, max(x2_todos) + margem_x2

    X1_grid, X2_grid = np.meshgrid(
        np.linspace(x1_min, x1_max, 100),
        np.linspace(x2_min, x2_max, 100)
    )
    Z_grid = funcao_alvo([X1_grid, X2_grid])

    ax1.contour(X1_grid, X2_grid, Z_grid, levels=40, cmap="viridis")

    cores = ["red", "blue", "green", "orange", "purple"]

    for i, res in enumerate(lista_resultados):
        cor = cores[i % len(cores)]
        nome = res["nome"]
        hist_x = res["hist_x"]
        hist_f = res["hist_f"]

        ax1.plot(hist_x[:, 0], hist_x[:, 1], marker="o", color=cor,
                 linestyle="-", label=nome, markersize=4)
        if i == 0:
            ax1.plot(hist_x[0, 0], hist_x[0, 1], marker="s",
                     color="black", markersize=8, label="Início")
        ax1.plot(hist_x[-1, 0], hist_x[-1, 1], marker="*",
                 color=cor, markersize=10)

        ax2.plot(range(len(hist_f)), hist_f, marker="o", color=cor,
                 linestyle="-", label=nome, markersize=4)

        if ax3 is not None and "hist_g" in res:
            ax3.plot(range(len(res["hist_g"])), res["hist_g"], marker="o",
                     color=cor, linestyle="-", label=nome, markersize=4)

        if ax3 is not None and "hist_dx" in res:
            ax3.plot(range(len(res["hist_dx"])), res["hist_dx"], marker="o",
                     color=cor, linestyle="-", label=nome, markersize=4)

    ax1.set_title("Curvas de Nível e Deslocamento")
    ax1.set_xlabel("x₁")
    ax1.set_ylabel("x₂")
    ax1.legend()

    ax2.set_title("Convergência — Iteração vs f(x₁, x₂)")
    ax2.set_xlabel("Iterações")
    ax2.set_ylabel("f(x₁, x₂)")
    ax2.legend()
    ax2.grid(True)

    if ax3 is not None:
        if tem_grad:
            ax3.set_title("Convergência — Iteração vs ‖∇f(x)‖")
            ax3.set_ylabel("‖∇f(x)‖")
        else:
            ax3.set_title("Convergência — Iteração vs ‖x_novo − x_ant‖")
            ax3.set_ylabel("‖x_novo − x_ant‖")
        ax3.set_xlabel("Iterações")
        ax3.legend()
        ax3.grid(True)

    plt.tight_layout()
    plt.show()