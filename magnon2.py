import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1) Construção da matriz pentadiagonal
# ============================================================

def build_matrix(kz, w, d0, d1_sup, d1_inf, d2_sup, d2_inf):
    n = len(d0)

    assert len(d1_sup) == n - 1
    assert len(d1_inf) == n - 1
    assert len(d2_sup) == n - 2
    assert len(d2_inf) == n - 2

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


# ============================================================
# 2) Cálculo direto das bandas
# ============================================================

def calcular_bandas_diretas(kz, diag, eta):
    """
    Calcula as frequências diretamente, sem busca de raízes.

    Válido quando:
        M(kz, w) = M0(kz) + eta*w*I

    com eta igual para todos os elementos da diagonal
    principal e nenhuma outra dependência em w.
    """

    # Constrói M0(kz), fazendo w = 0
    M0 = build_matrix(kz, 0.0, *diag)

    # Verifica se a matriz é simétrica
    if not np.allclose(M0, M0.T):
        raise ValueError(
            "A matriz não é simétrica. "
            "Este código usa np.linalg.eigvalsh e "
            "pressupõe diagonais superior e inferior iguais."
        )

    # Autovalores de M0, em ordem crescente
    mu = np.linalg.eigvalsh(M0)

    # Frequências autoconsistentes
    omega = mu / (1.0 - eta)

    return omega


# ============================================================
# 3) Caminho no espaço recíproco
# ============================================================

def caminho_kz(b, n_pontos=2000, kz_min=0.0, kz_max=None):

    if kz_max is None:
        kz_max = np.pi / b

    kz = np.linspace(kz_min, kz_max, n_pontos)

    return kz


# ============================================================
# 4) Cálculo de todas as bandas ao longo do caminho
# ============================================================

def calcular_dispersao(n_bandas, diag, b, eta,
                       n_pontos=2000,
                       kz_min=0.0,
                       kz_max=None):

    kz = caminho_kz(
        b=b,
        n_pontos=n_pontos,
        kz_min=kz_min,
        kz_max=kz_max
    )

    bandas = np.zeros((len(kz), n_bandas))

    for i, k in enumerate(kz):

        bandas[i, :] = calcular_bandas_diretas(
            k, diag, eta
        )

    return kz, bandas


# ============================================================
# 5) Gráfico da relação de dispersão
# ============================================================

def plot_dispersao(kz, bandas, b, passo_tick=0.25):

    x = kz * b / np.pi

    fig, ax = plt.subplots(figsize=(7, 5))

    for banda in range(bandas.shape[1]):
        ax.plot(
            x,
            bandas[:, banda],
            lw=1.8,
            label=f"Banda {banda + 1}"
        )

    ticks = np.arange(
        np.ceil(x.min() / passo_tick) * passo_tick,
        x.max() + 1e-9,
        passo_tick
    )

    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])

    ax.set_xlim(x.min(), x.max())

    ax.set_xlabel(r"$k_z b/\pi$")
    ax.set_ylabel(r"$\omega$")

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    return fig, ax


# ============================================================
# 6) Parâmetros do sistema
# ============================================================

if __name__ == "__main__":

    # Constante de rede
    b = 1.0

    # Coeficiente da dependência linear em omega
    eta = 0.001

    # Número de bandas
    n_bandas = 5

    # --------------------------------------------------------
    # Diagonal principal
    # --------------------------------------------------------

    A = [-2.77e-2] * n_bandas

    d0 = [
        lambda kz, w, a=a:
            a + 1.25e-2 * np.cos(kz * b) + eta * w
        for a in A
    ]

    # --------------------------------------------------------
    # Primeiras diagonais superior e inferior
    # --------------------------------------------------------

    J1_sup = [1.85e-2] * (n_bandas - 1)
    J1_inf = [1.85e-2] * (n_bandas - 1)

    d1_sup = [
        lambda kz, w, j=j: j
        for j in J1_sup
    ]

    d1_inf = [
        lambda kz, w, j=j: j
        for j in J1_inf
    ]

    # --------------------------------------------------------
    # Segundas diagonais superior e inferior
    # --------------------------------------------------------

    J2_sup = [1.06e-2] * (n_bandas - 2)
    J2_inf = [1.06e-2] * (n_bandas - 2)

    d2_sup = [
        lambda kz, w, j=j: j
        for j in J2_sup
    ]

    d2_inf = [
        lambda kz, w, j=j: j
        for j in J2_inf
    ]

    # Todas as diagonais
    diag = (
        d0,
        d1_sup,
        d1_inf,
        d2_sup,
        d2_inf
    )

    # --------------------------------------------------------
    # Calcula a relação de dispersão
    # --------------------------------------------------------

    kz, bandas = calcular_dispersao(
        n_bandas=n_bandas,
        diag=diag,
        b=b,
        eta=eta,
        n_pontos=2000,
        kz_min=0.0,
        kz_max=np.pi / b
    )

    # --------------------------------------------------------
    # Exibe os resultados
    # --------------------------------------------------------

    print("Frequências no primeiro ponto, kz = 0:")
    print(bandas[0])

    print("\nFrequências no último ponto, kz = pi/b:")
    print(bandas[-1])

    fig, ax = plot_dispersao(
        kz,
        bandas,
        b,
        passo_tick=0.25
    )

    plt.show()