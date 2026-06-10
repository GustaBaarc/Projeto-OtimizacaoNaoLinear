# Projeto — Otimização Não Linear

Projeto desenvolvido para a disciplina de Otimização Não Linear.  
Implementa e compara múltiplos métodos de otimização para funções de duas variáveis.

---

## Como rodar

```bash
python interface.py
```

Na janela que abrir:
1. Selecione um ou mais métodos
2. Informe o ponto inicial (x, y)
3. Defina o epsilon e o máximo de iterações
4. Escolha o critério de parada
5. Clique em **EXECUTAR**

Os gráficos e o histórico de iterações serão exibidos automaticamente.

---

## Estrutura dos arquivos

| Arquivo | O que faz |
|---|---|
| `interface.py` | Interface gráfica principal — é aqui que você configura e roda |
| `funcoes.py` | Todas as funções, gradientes e hessianas disponíveis |
| `gradiente.py` | Método do Vetor Gradiente (com busca de linha Seção Áurea) |
| `otimizacao_newton.py` | Método de Newton puro |
| `otimizacao_newton_modificado.py` | Método de Newton Modificado (com busca de linha) |
| `otimizacao_quasi_newton.py` | Método Quasi-Newton BFGS (com busca de linha) |
| `diraleatoria.py` | Busca Aleatória de direção |
| `criterios_parada.py` | Lógica dos 4 critérios de parada |
| `plotar.py` | Geração dos gráficos |
| `main.py` | Versão alternativa sem interface gráfica (editar direto no código) |

---

## Funções disponíveis

Definidas em `funcoes.py`. Para trocar a função ativa, edite a linha no topo de `interface.py`:

```python
funcao_ativa = "problema_a"  # altere aqui
```

| Nome | Função matemática | Mínimo global |
|---|---|---|
| `"f1"` | Booth | x* = (1, 3), f* = 0 |
| `"f4"` | Three-Hump Camel | x* = (0, 0), f* = 0 |
| `"rosenbrock"` | Rosenbrock | x* = (1, 1), f* = 0 |
| `"dixon"` | Dixon-Price | x* = (1, −0.7071), f* = 0 |
| `"problema_a"` | Problema A (matrículas) | x* = (65.2, 60.8), f* = 150 |

---

## Como adicionar uma função nova

### 1. Abrir `funcoes.py` e adicionar três blocos

```python
# ==========================================
# FUNÇÃO X: NOME DA FUNÇÃO
# ==========================================
def f_minha_funcao(x):
    return ...  # expressão da função

def grad_minha_funcao(x):
    return np.array([
        ...,  # derivada parcial em x[0]
        ...,  # derivada parcial em x[1]
    ])

def hess_minha_funcao(x):
    return np.array([
        [H00, H01],  # segunda linha da Hessiana
        [H10, H11],
    ])
```

> A Hessiana só é usada pelos métodos Newton e Newton Modificado.  
> Se for usar apenas Gradiente, Quasi-Newton ou Busca Aleatória, pode deixar uma Hessiana identidade.

### 2. Registrar em `interface.py`

Localize o bloco de `if/elif` que define `f, grad, hess` e adicione:

```python
elif funcao_ativa == "minha_funcao":
    f, grad, hess = f_minha_funcao, grad_minha_funcao, hess_minha_funcao
```

### 3. Trocar a função ativa

```python
funcao_ativa = "minha_funcao"
```

---

## Critérios de parada disponíveis

| Critério | Quando para |
|---|---|
| `delta_f` | Quando a variação relativa de f nas últimas 6 iterações for menor que ε |
| `norma_x` | Quando ‖x_novo − x_ant‖ < ε (convergência das variáveis) |
| `norma_grad` | Quando ‖∇f(x)‖ < ε (convergência do gradiente) |
| `max_iter` | Após um número fixo de iterações |

---

## O que é adicionado automaticamente ao histórico de cada método

| Método | hist_x | hist_f | hist_g (gradiente) | hist_dx (variáveis) |
|---|---|---|---|---|
| Gradiente | ✅ | ✅ | ✅ | — |
| Newton | ✅ | ✅ | — | — |
| Newton Modificado | ✅ | ✅ | — | — |
| Quasi-Newton | ✅ | ✅ | — | ✅ |
| Busca Aleatória | ✅ | ✅ | — | — |

O `plotar.py` detecta automaticamente quais históricos estão disponíveis e exibe o 3º gráfico apenas quando `hist_g` ou `hist_dx` existirem.

---

## Observações importantes

- O ponto inicial **deve ser informado na interface** a cada execução — não há valor salvo automaticamente
- A função ativa **precisa ser trocada manualmente** em `interface.py` antes de rodar
- A busca de linha usa Seção Áurea com `b=2.0` como limite de passo — funções com ótimo muito distante da origem podem precisar aumentar esse valor nos respectivos arquivos
- Para rodar sem interface gráfica, edite `main.py` diretamente e execute `python main.py`