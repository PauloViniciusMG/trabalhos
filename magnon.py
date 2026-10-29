"""
Relação de dispersão w(kz) a partir de uma matriz pentadiagonal cujas
entradas são funções de (kz, w) -- problema de autovalor NAO-LINEAR.

A rede é infinita em UMA única direção: z (eixo b, constante de rede b).
kz é escalar; kx não entra (direção finita).

Entradas da matriz (5 listas de funções f(kz, w), uma função por posição):
    d0     : n   termos -> diagonal principal      M[i, i]
    d1_sup : n-1 termos -> diagonal +1 (acima)     M[i, i+1]
    d1_inf : n-1 termos -> diagonal -1 (abaixo)    M[i+1, i]
    d2_sup : n-2 termos -> diagonal +2 (acima)     M[i, i+2]
    d2_inf : n-2 termos -> diagonal -2 (abaixo)    M[i+2, i]
As diagonais superior e inferior NAO precisam ser iguais.

Como a matriz pode ser assimétrica, usa-se np.linalg.eigvals (e não eigvalsh).
Os autovalores são ordenados pela parte real; o código trabalha com Re(w) e
avisa se a parte imaginária dos autovalores for significativa.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

TOL_IMAG = 1e-8


# ---------------------------------------------------------------------
# 1) Construção da matriz pentadiagonal (sup e inf independentes)
# ---------------------------------------------------------------------
def build_matrix(kz, w, d0, d1_sup, d1_inf, d2_sup, d2_inf):
    n = len(d0)
    assert len(d1_sup) == n - 1 and len(d1_inf) == n - 1
    assert len(d2_sup) == n - 2 and len(d2_inf) == n - 2

    M = np.zeros((n, n))
    for i in range(n):
        M[i, i] = d0[i](kz, w)
    for i in range(n - 1):
        M[i, i + 1] = d1_sup[i](kz, w)
        M[i + 1, i] = d1_inf[i](kz, w)
    for i in range(n - 2):
        M[i, i + 2] = d2_sup[i](kz, w)
        M[i + 2, i] = d2_inf[i](kz, w)
    return M


def autovalores_ordenados(M):
    """Autovalores de M (matriz geral), ordenados pela parte real."""
    ev = np.linalg.eigvals(M)
    return ev[np.argsort(ev.real)]


# ---------------------------------------------------------------------
# 2) Solução autoconsistente: para a banda b, achar w tal que
#    Re(autovalor_b(M(kz,w))) == w
# ---------------------------------------------------------------------
def _residuo_banda(w, kz, banda, diag):
    M = build_matrix(kz, w, *diag)
    ev = autovalores_ordenados(M)[banda]
    if abs(ev.imag) > TOL_IMAG:
        warnings.warn(f"autovalor complexo (Im={ev.imag:.2e}) em kz={kz:.3f}, "
                      f"w={w:.3f}; usando só a parte real", stacklevel=2)
    return ev.real - w


def resolver_banda(kz, banda, diag, w_min, w_max,
                   chute=None, n_scan=150, janela_chute=3.0):
    """
    Valor de w autoconsistente para uma banda. Com 'chute' (continuação do
    kz anterior) tenta primeiro uma janela estreita; se falhar, varre
    [w_min, w_max].
    """
    def g(w):
        return _residuo_banda(w, kz, banda, diag)

    if chute is not None and not np.isnan(chute):
        ws = np.linspace(chute - janela_chute, chute + janela_chute, 25)
        gs = [g(w) for w in ws]
        for i in range(len(ws) - 1):
            if gs[i] == 0:
                return ws[i]
            if gs[i] * gs[i + 1] < 0:
                return brentq(g, ws[i], ws[i + 1], xtol=1e-9)

    ws = np.linspace(w_min, w_max, n_scan)
    gs = [g(w) for w in ws]
    for i in range(len(ws) - 1):
        if gs[i] == 0:
            return ws[i]
        if gs[i] * gs[i + 1] < 0:
            return brentq(g, ws[i], ws[i + 1], xtol=1e-9)

    return np.nan


def resolver_todas_bandas(kz, n_bandas, diag, w_min, w_max, chutes=None):
    resultado = np.full(n_bandas, np.nan)
    for b in range(n_bandas):
        chute_b = None if chutes is None else chutes[b]
        resultado[b] = resolver_banda(kz, b, diag, w_min, w_max, chute=chute_b)
    return resultado


# ---------------------------------------------------------------------
# 3) Caminho 1D: kz de -pi/b até +pi/b (1a zona de Brillouin da direção
#    infinita z). Use kz_min=0 para só 0 -> pi/b.
# ---------------------------------------------------------------------
def caminho_kz(b, n_pontos=200, kz_min=None, kz_max=None):
    if kz_min is None:
        kz_min = -np.pi / b
    if kz_max is None:
        kz_max = np.pi / b
    kz = np.linspace(kz_min, kz_max, n_pontos)
    marcos = [kz_min, 0.0, kz_max]
    return kz, marcos


# ---------------------------------------------------------------------
# 4) Varre o caminho, resolvendo a autoconsistência em cada kz
#    (resultado do kz anterior como chute -> bandas contínuas)
# ---------------------------------------------------------------------
def varrer_caminho(n_bandas, diag, b, w_min, w_max, n_pontos=200,
                   kz_min=None, kz_max=None):
    kz, marcos = caminho_kz(b, n_pontos, kz_min, kz_max)
    bandas = np.full((len(kz), n_bandas), np.nan)

    chutes = None
    for i in range(len(kz)):
        bandas[i] = resolver_todas_bandas(kz[i], n_bandas, diag,
                                          w_min, w_max, chutes=chutes)
        chutes = bandas[i]
    return kz, bandas, marcos


# ---------------------------------------------------------------------
# 5) Plot 1D: uma curva por banda, eixo x em kz*b/pi
# ---------------------------------------------------------------------
def plot_dispersao(kz, bandas, b, passo_tick=0.25):
    x = kz * b / np.pi                      # kz*b/pi, vai de -1 a +1

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for banda in range(bandas.shape[1]):
        ax.plot(x, bandas[:, banda], lw=1.8)

    # ticks espaçados ao longo do eixo (ex.: -1, -0.75, ..., 1)
    ticks = np.arange(np.ceil(x.min() / passo_tick) * passo_tick,
                      x.max() + 1e-9, passo_tick)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])

    ax.set_xlim(x.min(), x.max())
    ax.set_xlabel(r"$k_z b/\pi$")
    ax.set_ylabel(r"$\omega$")
    ax.grid(True, alpha=0.3)
    #ax.set_title(r"Dispersão autoconsistente 1D: $\omega$ = autovalor de $M(k_z,\omega)$")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------
# 6) Verificação: sem dependência em w, a solução autoconsistente deve
#    coincidir com a diagonalização direta
# ---------------------------------------------------------------------
def verificar_caso_linear(b=1.0):
    n = 5
    A = [1.0, 1.3, 0.9, 1.5, 1.1]
    J1s = [-0.6, -0.4, -0.5, -0.3]; J1i = [-0.5, -0.45, -0.55, -0.35]  # sup != inf
    J2s = [-0.15, -0.10, -0.05];    J2i = [-0.12, -0.08, -0.06]        # sup != inf

    d0     = [(lambda kz, w, a=a: a + 0.5 * np.cos(kz * b)) for a in A]
    d1_sup = [(lambda kz, w, j=j: j) for j in J1s]
    d1_inf = [(lambda kz, w, j=j: j) for j in J1i]
    d2_sup = [(lambda kz, w, j=j: j) for j in J2s]
    d2_inf = [(lambda kz, w, j=j: j) for j in J2i]
    diag = (d0, d1_sup, d1_inf, d2_sup, d2_inf)

    kz_teste = 0.7
    M = build_matrix(kz_teste, 0.0, *diag)
    diretos = np.sort(np.linalg.eigvals(M).real)
    auto = resolver_todas_bandas(kz_teste, n, diag, w_min=-5, w_max=5)

    erro = np.nanmax(np.abs(diretos - np.sort(auto)))
    print(f"[verificação sem dependência em w] erro máx.: {erro:.2e}")


# ---------------------------------------------------------------------
# 7) Exemplo com dependência real em w
# ---------------------------------------------------------------------
if __name__ == "__main__":

    # constante de rede na direção infinita (z): definida UMA vez e usada
    # tanto nas funções (fase de Bloch kz*b) quanto no caminho em kz
    b = 1

    verificar_caso_linear(b)

    n_bandas = 5

    # diagonal principal: valores diferentes + termo linear em w
    A   = [ 4.155 * 10**(-2), 4.155 * 10**(-2), 4.155 * 10**(-2), 4.155 * 10**(-2), 4.155 * 10**(-2)]
    ETA = [ 0.001, 0.001, 0.001, 0.001, 0.001]
    d0 = [(lambda kz, w, a=a, eta=eta: a - (0.625*10**(-2)) * np.cos(kz * b) + eta * w)
          for a, eta in zip(A, ETA)]

    # diagonais ±1: superior e inferior DIFERENTES (sem dependência em w)
    J1_sup = [ -9.24 * 10**(-3), -9.24 * 10**(-3), -9.24 * 10**(-3), -9.24 * 10**(-3)]
    J1_inf = [ -9.24 * 10**(-3), -9.24 * 10**(-3), -9.24 * 10**(-3), -9.24 * 10**(-3)]
    d1_sup = [(lambda kz, w, j=j: j) for j in J1_sup]
    d1_inf = [(lambda kz, w, j=j: j) for j in J1_inf]

    # diagonais ±2: superior e inferior DIFERENTES
    J2_sup = [-5.31*10**(-3), -5.31*10**(-3), -5.31*10**(-3)]
    J2_inf = [ -5.31*10**(-3), -5.31*10**(-3), -5.31*10**(-3),]
    d2_sup = [(lambda kz, w, j=j: j) for j in J2_sup]
    d2_inf = [(lambda kz, w, j=j: j) for j in J2_inf]

    diag = (d0, d1_sup, d1_inf, d2_sup, d2_inf)

    # kz de 0 a +pi/b  (kz*b/pi de 0 a 1)
    kz, bandas, marcos = varrer_caminho(n_bandas, diag, b=b,
                                        w_min=-0.2, w_max=0.2, n_pontos=2000,
                                        kz_min=0.0)

    # eixo x em kz*b/pi, ticks a cada 0.25 (mude com passo_tick=...)
    fig, ax = plot_dispersao(kz, bandas, b, passo_tick=0.25)
    plt.show()