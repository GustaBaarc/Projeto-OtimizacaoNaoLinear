import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from funcoes import (
    f1, grad1, hess1,
    f4, grad4, hess4,
    f_rosenbrock, grad_rosenbrock, hess_rosenbrock,
    f_dixon, grad_dixon, hess_dixon,
    f_problema_a, grad_problema_a, hess_problema_a,
)
from gradiente import rodar_gradiente
from otimizacao_newton import rodar_newton
from otimizacao_quasi_newton import rodar_quasi_newton
from diraleatoria import rodar_aleatoria
from plotar import exibir_graficos
from otimizacao_newton_modificado import rodar_newton_modificado

# ──────────────────────────────────────────────────────────────────────────────
# ESCOLHA A FUNÇÃO ATIVA AQUI
#   "f1"         → Booth
#   "f4"         → Three-Hump Camel   
#   "rosenbrock" → Rosenbrock         
#   "dixon"      → Dixon-Price        
#   "problema_a" → Problema A         
# ──────────────────────────────────────────────────────────────────────────────
funcao_ativa = "problema_a"

if funcao_ativa == "f1":
    f, grad, hess = f1, grad1, hess1
elif funcao_ativa == "f4":
    f, grad, hess = f4, grad4, hess4
elif funcao_ativa == "rosenbrock":
    f, grad, hess = f_rosenbrock, grad_rosenbrock, hess_rosenbrock
elif funcao_ativa == "dixon":
    f, grad, hess = f_dixon, grad_dixon, hess_dixon
elif funcao_ativa == "problema_a":
    f, grad, hess = f_problema_a, grad_problema_a, hess_problema_a
else:
    raise ValueError("Função desconhecida.")


BG_DARK, BG_CARD, ACCENT, TEXT_MAIN = "#0f1117", "#1a1d27", "#4f8ef7", "#e8eaf6"

CRITERIOS_DISPONIVEIS = ["delta_f", "norma_x", "norma_grad", "max_iter"]
CRITERIO_DESCRICAO = {
    "delta_f":    "Δf relativo (últimas 6 iter.)",
    "norma_x":    "‖x_novo − x_ant‖ < ε  (variáveis)",
    "norma_grad": "‖∇f(x)‖ < ε  (gradiente)",
    "max_iter":   "N iterações fixas",
}


def rodar_tudo():
    try:
        p0 = [float(campo_x.get()), float(campo_y.get())]
        max_iter = int(campo_iter.get())
        eps = float(campo_eps.get())
        criterio = criterio_var.get()
        max_iter_fixo = max_iter if criterio == "max_iter" else None

        lista_resultados = []

        print("\n" + "=" * 50)
        print(f"  FUNÇÃO: {funcao_ativa.upper()}")
        print(f"  CRITÉRIO DE PARADA: {criterio}")
        print(f"  X0 = {p0}   ε = {eps}")
        print("=" * 50)

        if chk_gradiente.get():
            c_x, c_f, c_g = rodar_gradiente(f, grad, p0, max_iter, eps,
                                       criterio=criterio, max_iter_fixo=max_iter_fixo)
            lista_resultados.append({"nome": "Gradiente", "hist_x": c_x, "hist_f": c_f, "hist_g": c_g})

        if chk_newton.get():
            c_x, c_f = rodar_newton(f, grad, hess, p0, max_iter, eps,
                                    criterio=criterio, max_iter_fixo=max_iter_fixo)
            lista_resultados.append({"nome": "Newton", "hist_x": c_x, "hist_f": c_f})

        if chk_bfgs.get():
            c_x, c_f = rodar_quasi_newton(f, grad, p0, max_iter, eps,
                                          criterio=criterio, max_iter_fixo=max_iter_fixo)
            lista_resultados.append({"nome": "Quasi-Newton", "hist_x": c_x, "hist_f": c_f})

        if chk_aleatoria.get():
            c_x, c_f = rodar_aleatoria(f, p0, max_iter, eps,
                                       criterio=criterio, max_iter_fixo=max_iter_fixo)
            lista_resultados.append({"nome": "Aleatória", "hist_x": c_x, "hist_f": c_f})

        if chk_newton_mod.get():
            c_x, c_f = rodar_newton_modificado(f, grad, hess, p0, max_iter, eps,
                                               criterio=criterio, max_iter_fixo=max_iter_fixo)
            lista_resultados.append({"nome": "Newton Mod.", "hist_x": c_x, "hist_f": c_f})

        if not lista_resultados:
            messagebox.showwarning("Aviso", "Selecione pelo menos um método!")
            return

        print("\n" + "=" * 50)
        print("  HISTÓRICO DE ITERAÇÕES")
        print("=" * 50)
        for res in lista_resultados:
            print(f"\n--- {res['nome'].upper()} ---")
            for i in range(len(res["hist_f"])):
                print(
                    f"  it {i:>3}: x=[{res['hist_x'][i][0]:.6f}, {res['hist_x'][i][1]:.6f}]"
                    f"  f={res['hist_f'][i]:.8f}"
                )
            x_final = res["hist_x"][-1]
            print(f"  → Solução: x*=[{x_final[0]:.6f}, {x_final[1]:.6f}]"
                  f"  f*={res['hist_f'][-1]:.8f}"
                  f"  ({len(res['hist_f'])-1} iterações)")

        exibir_graficos(lista_resultados, f)

    except Exception as e:
        messagebox.showerror("Erro", str(e))
        raise


# ──────────────────────────────────────────────────────────────────────────────
# INTERFACE
# ──────────────────────────────────────────────────────────────────────────────
janela = tk.Tk()
janela.title("Otimização ONL-I")
janela.geometry("430x620")
janela.configure(bg=BG_DARK, padx=20, pady=20)

tk.Label(janela, text=f"FUNÇÃO ATIVA: {funcao_ativa.upper()}",
         bg=BG_DARK, fg=ACCENT, font=("Consolas", 10, "bold")).pack(pady=5)

card = tk.Frame(janela, bg=BG_CARD, padx=15, pady=15,
                highlightthickness=1, highlightbackground="#2a2d3e")
card.pack(fill="both", expand=True)

var_m = [tk.BooleanVar() for _ in range(5)]
chk_gradiente, chk_newton, chk_bfgs, chk_aleatoria, chk_newton_mod = var_m
labels = ["Vetor Gradiente", "Newton", "Quasi-Newton (BFGS)", "Busca Aleatória", "Newton Modificado"]
for i, txt in enumerate(labels):
    tk.Checkbutton(card, text=txt, variable=var_m[i],
                   bg=BG_CARD, fg=TEXT_MAIN, selectcolor=BG_DARK).pack(anchor="w")

tk.Label(card, text="\nPONTO INICIAL (x, y):", bg=BG_CARD, fg="#6c7293",
         font=("Consolas", 8)).pack(anchor="w")
f_p = tk.Frame(card, bg=BG_CARD)
f_p.pack(fill="x")
campo_x = tk.Entry(f_p, width=10)
campo_x.insert(0, 0)
campo_x.pack(side="left", padx=5)
campo_y = tk.Entry(f_p, width=10)
campo_y.insert(0, 0)
campo_y.pack(side="left")

tk.Label(card, text="MAX ITERAÇÕES:", bg=BG_CARD, fg="#6c7293").pack(anchor="w", pady=(10, 0))
campo_iter = tk.Entry(card)
campo_iter.insert(0, "50")
campo_iter.pack(fill="x")

tk.Label(card, text="EPSILON:", bg=BG_CARD, fg="#6c7293").pack(anchor="w", pady=(10, 0))
campo_eps = tk.Entry(card)
campo_eps.insert(0, "0.001")
campo_eps.pack(fill="x")

tk.Label(card, text="\nCRITÉRIO DE PARADA:", bg=BG_CARD, fg=ACCENT,
         font=("Consolas", 8, "bold")).pack(anchor="w")
criterio_var = tk.StringVar(value="delta_f")
for c in CRITERIOS_DISPONIVEIS:
    tk.Radiobutton(
        card, text=f"{c}  —  {CRITERIO_DESCRICAO[c]}",
        variable=criterio_var, value=c,
        bg=BG_CARD, fg=TEXT_MAIN, selectcolor=BG_DARK,
        font=("Consolas", 8),
    ).pack(anchor="w")

btn = tk.Button(card, text="EXECUTAR", bg=ACCENT, fg="white",
                font=("Consolas", 10, "bold"), command=rodar_tudo)
btn.pack(fill="x", pady=20, ipady=5)

janela.mainloop()