import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from funcoes import f1, grad1, hess1, f4, grad4, hess4
from gradiente import rodar_gradiente
from otimizacao_newton import rodar_newton
from otimizacao_quasi_newton import rodar_quasi_newton
from diraleatoria import rodar_aleatoria
from plotar import exibir_graficos
from otimizacao_newton_modificado import rodar_newton_modificado

funcao_ativa = "f1"

if funcao_ativa == "f1":
    f = f1
    grad = grad1
    hess = hess1
    fundo = f1
elif funcao_ativa == "f4":
    f = f4
    grad = grad4
    hess = hess4
    fundo = f4
else:
    raise ValueError("Coloca f1 ou f4 só!")


# -------------------------------------------------------
# Cores e fontes do tema escuro
# -------------------------------------------------------
BG_DARK = "#0f1117"
BG_CARD = "#1a1d27"
BG_HOVER = "#22263a"
ACCENT = "#4f8ef7"
ACCENT2 = "#7c5bf7"
TEXT_MAIN = "#e8eaf6"
TEXT_DIM = "#6c7293"
TEXT_WHITE = "#ffffff"
BORDER = "#2a2d3e"
GREEN = "#3ecf8e"
RED_WARN = "#f7594f"

FONT_TITULO = ("Consolas", 13, "bold")
FONT_LABEL = ("Consolas", 9)
FONT_ENTRY = ("Consolas", 10)
FONT_CHECK = ("Consolas", 9)
FONT_BTN = ("Consolas", 10, "bold")
FONT_SECAO = ("Consolas", 8, "bold")


def rodar_tudo():
    try:
        px = float(campo_x.get())
        py = float(campo_y.get())
        p0 = [px, py]

        max_iter = int(campo_iter.get())
        alpha = float(campo_alpha.get())
        eps = float(campo_eps.get())

        alfas_bfgs = [0.4125, 0.7530]
        lista_resultados = []
        fundo_grafico = f

        if not (
            chk_gradiente.get()
            or chk_newton.get()
            or chk_bfgs.get()
            or chk_aleatoria.get()
            or chk_newton_mod.get()
        ):
            messagebox.showwarning("Ops!", "Marca pelo menos um método pra rodar!")
            return

        print("\n" + "=" * 45)
        print("  RODANDO OTIMIZAÇÃO...")
        print("=" * 45)

        # --- Gradiente ---
        if chk_gradiente.get():
            print("[1] Rodando Gradiente...")
            caminho_x, caminho_f = rodar_gradiente(f, grad, p0, max_iter, alpha, eps)
            lista_resultados.append(
                {"nome": "Gradiente", "hist_x": caminho_x, "hist_f": caminho_f}
            )

        # --- Newton ---
        if chk_newton.get():
            print("[2] Rodando Newton...")
            caminho_x, caminho_f = rodar_newton(f, grad, hess, p0, max_iter, eps)
            lista_resultados.append(
                {"nome": "Newton", "hist_x": caminho_x, "hist_f": caminho_f}
            )

        # --- Quasi-Newton  ---
        if chk_bfgs.get():
            print("[3] Rodando Quasi-Newton...")
            caminho_x, caminho_f = rodar_quasi_newton(
                f, grad, p0, max_iter, alfas_bfgs, alpha, eps
            )
            lista_resultados.append(
                {"nome": "Quasi-Newton", "hist_x": caminho_x, "hist_f": caminho_f}
            )

        # --- Busca Aleatoria ---
        if chk_aleatoria.get():
            print("[4] Rodando Busca Aleatória...")
            caminho_x, caminho_f = rodar_aleatoria(f, p0, max_iter, None, 0.1, eps)
            lista_resultados.append(
                {"nome": "Busca Aleatória", "hist_x": caminho_x, "hist_f": caminho_f}
            )

        # --- Newton Modificado ---
        if chk_newton_mod.get():
            print("[5] Rodando Newton Modificado...")
            caminho_x, caminho_f = rodar_newton_modificado(
                f, grad, hess, p0, max_iter, alpha, eps
            )
            lista_resultados.append(
                {"nome": "Newton Modificado", "hist_x": caminho_x, "hist_f": caminho_f}
            )

        # Printa historico no terminal
        print("\n" + "=" * 45)
        print("  HISTÓRICO DE ITERAÇÕES")
        print("=" * 45)
        for res in lista_resultados:
            print(f"\n--- {res['nome'].upper()} ---")
            for i in range(len(res["hist_f"])):
                print(
                    f"  it {i:>3}: x=[{res['hist_x'][i][0]:.5f}, {res['hist_x'][i][1]:.5f}]"
                    f"  f={res['hist_f'][i]:.6f}"
                )

        # Manda pro grafico
        exibir_graficos(lista_resultados, fundo_grafico)

    except ValueError:
        messagebox.showerror("Erro", "Algum campo tá com valor inválido!")
    except Exception as erro:
        messagebox.showerror("Erro inesperado", str(erro))


# -------------------------------------------------------
# Funcoes auxiliares de estilo
# -------------------------------------------------------
def criar_entry_estilizado(pai, valor_padrao, largura=8):
    entry = tk.Entry(
        pai,
        width=largura,
        font=FONT_ENTRY,
        bg=BG_DARK,
        fg=TEXT_MAIN,
        insertbackground=ACCENT,
        relief="flat",
        bd=0,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=ACCENT,
    )
    entry.insert(0, valor_padrao)
    return entry


def criar_label(pai, texto, fonte=FONT_LABEL, cor=TEXT_DIM):
    return tk.Label(pai, text=texto, font=fonte, bg=BG_CARD, fg=cor)


def criar_separador(pai):
    sep = tk.Frame(pai, bg=BORDER, height=1)
    sep.pack(fill="x", pady=8)
    return sep


def ao_passar_mouse(event, btn, dentro):
    btn.config(bg=BG_HOVER if dentro else BG_CARD)


# -------------------------------------------------------
# Janela principal
# -------------------------------------------------------
janela = tk.Tk()
janela.title("Otimização Não-Linear")
janela.configure(bg=BG_DARK)
janela.resizable(False, False)

# Centraliza a janela
larg, alt = 430, 580
x_pos = janela.winfo_screenwidth() // 2 - larg // 2
y_pos = janela.winfo_screenheight() // 2 - alt // 2
janela.geometry(f"{larg}x{alt}+{x_pos}+{y_pos}")

# -------------------------------------------------------
# Header
# -------------------------------------------------------
header = tk.Frame(janela, bg=BG_DARK, pady=0)
header.pack(fill="x")

# Barra colorida no topo
barra_topo = tk.Canvas(header, height=4, bg=BG_DARK, highlightthickness=0)
barra_topo.pack(fill="x")
barra_topo.update_idletasks()
w = larg
barra_topo.create_rectangle(0, 0, w // 2, 4, fill=ACCENT, outline="")
barra_topo.create_rectangle(w // 2, 0, w, 4, fill=ACCENT2, outline="")

titulo_frame = tk.Frame(header, bg=BG_DARK, padx=20, pady=14)
titulo_frame.pack(fill="x")

tk.Label(
    titulo_frame,
    text="OTIMIZAÇÃO NÃO-LINEAR",
    font=("Consolas", 14, "bold"),
    bg=BG_DARK,
    fg=TEXT_WHITE,
).pack(anchor="w")

tk.Label(
    titulo_frame,
    text=f"função ativa: {funcao_ativa} ",
    font=("Consolas", 8),
    bg=BG_DARK,
    fg=TEXT_DIM,
).pack(anchor="w", pady=(2, 0))

# -------------------------------------------------------
# Card principal
# -------------------------------------------------------
card = tk.Frame(
    janela,
    bg=BG_CARD,
    padx=20,
    pady=18,
    highlightthickness=1,
    highlightbackground=BORDER,
)
card.pack(fill="both", expand=True, padx=14, pady=(0, 14))

# ---- Seção: Métodos ----
tk.Label(
    card,
    text="▸ MÉTODOS",
    font=FONT_SECAO,
    bg=BG_CARD,
    fg=ACCENT,
).pack(anchor="w", pady=(0, 6))

chk_gradiente = tk.BooleanVar(value=True)
chk_newton = tk.BooleanVar()
chk_bfgs = tk.BooleanVar()
chk_aleatoria = tk.BooleanVar()
chk_newton_mod = tk.BooleanVar()

metodos = [
    ("Vetor Gradiente", chk_gradiente),
    ("Método de Newton", chk_newton),
    ("Quasi-Newton  ", chk_bfgs),
    ("Busca Aleatória", chk_aleatoria),
    ("Newton Modificado", chk_newton_mod),
]

frame_checks = tk.Frame(card, bg=BG_CARD)
frame_checks.pack(fill="x", pady=(0, 4))

for nome, var in metodos:
    linha = tk.Frame(frame_checks, bg=BG_CARD)
    linha.pack(fill="x", pady=1)

    cb = tk.Checkbutton(
        linha,
        text=f"  {nome}",
        variable=var,
        font=FONT_CHECK,
        bg=BG_CARD,
        fg=TEXT_MAIN,
        selectcolor=BG_DARK,
        activebackground=BG_CARD,
        activeforeground=TEXT_WHITE,
        highlightthickness=0,
        bd=0,
        cursor="hand2",
    )
    cb.pack(anchor="w")

criar_separador(card)

# ---- Seção: Ponto inicial ----
tk.Label(
    card,
    text="▸ PONTO INICIAL",
    font=FONT_SECAO,
    bg=BG_CARD,
    fg=ACCENT,
).pack(anchor="w", pady=(0, 6))

frame_ponto = tk.Frame(card, bg=BG_CARD)
frame_ponto.pack(fill="x", pady=(0, 4))

criar_label(frame_ponto, "x₀ =").pack(side="left")
campo_x = criar_entry_estilizado(frame_ponto, "0.5")
campo_x.pack(side="left", padx=(4, 20))

criar_label(frame_ponto, "y₀ =").pack(side="left")
campo_y = criar_entry_estilizado(frame_ponto, "3.5")
campo_y.pack(side="left", padx=4)

criar_separador(card)

# ---- Seção: Parâmetros ----
tk.Label(
    card,
    text="▸ PARÂMETROS",
    font=FONT_SECAO,
    bg=BG_CARD,
    fg=ACCENT,
).pack(anchor="w", pady=(0, 8))

parametros = [
    ("Iterações máx.", "5", "campo_iter"),
    ("Alpha  (passo)", "0.05", "campo_alpha"),
    ("Epsilon (parada)", "0.001", "campo_eps"),
]

campos = {}
for label_txt, default, nome in parametros:
    linha = tk.Frame(card, bg=BG_CARD)
    linha.pack(fill="x", pady=4)

    tk.Label(
        linha,
        text=label_txt,
        font=FONT_LABEL,
        bg=BG_CARD,
        fg=TEXT_DIM,
        width=16,
        anchor="w",
    ).pack(side="left")

    entry = criar_entry_estilizado(linha, default, largura=14)
    entry.pack(side="left", padx=(6, 0), ipady=4)
    campos[nome] = entry

campo_iter = campos["campo_iter"]
campo_alpha = campos["campo_alpha"]
campo_eps = campos["campo_eps"]

criar_separador(card)

# ---- Botão de executar ----
btn_frame = tk.Frame(card, bg=BG_CARD)
btn_frame.pack(fill="x", pady=(4, 0))

btn = tk.Button(
    btn_frame,
    text="▶  EXECUTAR OTIMIZAÇÃO",
    font=FONT_BTN,
    bg=ACCENT,
    fg=TEXT_WHITE,
    activebackground=ACCENT2,
    activeforeground=TEXT_WHITE,
    relief="flat",
    bd=0,
    padx=12,
    pady=10,
    cursor="hand2",
    command=rodar_tudo,
)
btn.pack(fill="x", ipady=2)

btn.bind("<Enter>", lambda e: btn.config(bg=ACCENT2))
btn.bind("<Leave>", lambda e: btn.config(bg=ACCENT))

# ---- Rodapé ----
tk.Label(
    janela,
    text="UNIMONTES  ·  Otimização Não Linear  ·  2026",
    font=("Consolas", 10),
    bg=BG_DARK,
    fg=TEXT_DIM,
).pack(pady=(0, 8))

janela.mainloop()
