"""
Esempio didattico: effetto della risoluzione in energia di un rivelatore.

Lo spettro "vero" oscillante in E_true viene ricostruito con uno smearing
gaussiano la cui larghezza dipende dall'energia vera:

    p(E_rec | E_true) = G(E_rec; E_true, sigma(E_true))

Lo spettro ricostruito e' la convoluzione con kernel variabile:

    N_rec(E_rec) = Integrale dE_true  N_true(E_true) * p(E_rec | E_true)

Lo smearing non cambia il periodo dell'oscillazione, ma ne riduce l'ampiezza
di un fattore exp(-2 pi^2 sigma^2 / P^2), tanto piu' quanto peggiore e' la
risoluzione.

Figura a tre pannelli:
  - alto:    spettro in E_rec (dopo la risoluzione)
  - centro:  kernel analitico p(E_rec | E_true), E_rec (x) vs E_true (y)
  - destra:  spettro in E_true (prima della risoluzione)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ----------------------------------------------------------------------------
# Parametri
# ----------------------------------------------------------------------------
E_MIN, E_MAX = 1.0, 10.0   # intervallo di energia vera [MeV]
PERIODO = 1.0              # periodo dell'oscillazione vera [MeV]
AMPIEZZA = 0.8             # ampiezza relativa del seno (0 < A < 1)
A_STOC = 0.095             # termine stocastico: sigma/E = A_STOC / sqrt(E [MeV])
REC_MIN, REC_MAX = E_MIN, E_MAX   # intervallo mostrato in energia ricostruita [MeV]


def spettro_vero(E):
    """Spettro vero dN/dE_true (unita' arbitrarie): costante + seno."""
    return 1.0 + AMPIEZZA * np.sin(2 * np.pi * E / PERIODO)


def sigma(E):
    """Risoluzione gaussiana [MeV]: sigma/E = A_STOC/sqrt(E), cioe' sigma = A_STOC*sqrt(E)."""
    return A_STOC * np.sqrt(E)


def kernel(E_rec, E_true):
    """Densita' di probabilita' p(E_rec | E_true) [1/MeV]; restituisce una matrice (E_true, E_rec)."""
    s = sigma(E_true)[:, None]
    return np.exp(-0.5 * ((E_rec[None, :] - E_true[:, None]) / s) ** 2) / (np.sqrt(2 * np.pi) * s)


def main():
    E_true = np.linspace(E_MIN, E_MAX, 3000)
    E_rec = np.linspace(REC_MIN, REC_MAX, 4000)   # passo ~2.3 keV << sigma minima
    dE_true = E_true[1] - E_true[0]

    K = kernel(E_rec, E_true)
    N_true = spettro_vero(E_true)
    N_rec = N_true @ K * dE_true                  # convoluzione con kernel variabile

    # --- Figura --------------------------------------------------------------
    fig = plt.figure(figsize=(7.5, 7.5))
    gs = fig.add_gridspec(2, 2, width_ratios=(3, 1.2), height_ratios=(1.2, 3),
                          wspace=0.05, hspace=0.05)
    ax_c = fig.add_subplot(gs[1, 0])
    ax_t = fig.add_subplot(gs[0, 0], sharex=ax_c)
    ax_r = fig.add_subplot(gs[1, 1], sharey=ax_c)

    # Pannello centrale: kernel p(E_rec | E_true)
    Ks = K[::3]                                   # sottocampionamento in E_true per il disegno
    pdf_max = 1 / (np.sqrt(2 * np.pi) * sigma(E_MIN))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")                         # sfondo bianco dove il kernel e' trascurabile
    mesh = ax_c.pcolormesh(E_rec, E_true[::3], np.ma.masked_less(Ks, 1e-2), cmap=cmap,
                           norm=LogNorm(vmin=1e-2, vmax=pdf_max), shading="auto", rasterized=True)
    Eg = np.linspace(E_MIN, E_MAX, 200)
    ax_c.plot(Eg, Eg, color="0.2", lw=0.8, alpha=0.7, label=r"$E_{rec} = E_{true}$")
    ax_c.plot(Eg + sigma(Eg), Eg, color="0.2", lw=0.8, ls="--", label=r"$\pm1\sigma(E_{true})$")
    ax_c.plot(Eg - sigma(Eg), Eg, color="0.2", lw=0.8, ls="--")
    ax_c.set_xlabel(r"$E_{rec}$ [MeV]")
    ax_c.set_ylabel(r"$E_{true}$ [MeV]")
    ax_c.set_xlim(REC_MIN, REC_MAX)
    ax_c.set_ylim(E_MIN, E_MAX)
    ax_c.legend(loc="upper left")

    # Pannello in alto: spettro ricostruito
    ax_t.fill_between(E_rec, N_rec, color="tab:red", alpha=0.35, lw=0)
    ax_t.plot(E_rec, N_rec, color="tab:red", lw=1.2)
    ax_t.set_ylabel(r"$dN/dE_{rec}$ [u.a.]")
    ax_t.set_title("Spettro in energia ricostruita (dopo la risoluzione)")
    ax_t.tick_params(labelbottom=False)
    ax_t.set_ylim(0, 1.1 * (1 + AMPIEZZA))

    # Pannello a destra: spettro vero (ruotato)
    ax_r.fill_betweenx(E_true, N_true, color="tab:blue", alpha=0.35, lw=0)
    ax_r.plot(N_true, E_true, color="tab:blue", lw=1.2)
    ax_r.set_xlabel(r"$dN/dE_{true}$ [u.a.]")
    ax_r.set_title("Spettro in\nenergia vera", fontsize=10)
    ax_r.tick_params(labelleft=False)
    ax_r.set_xlim(0, 1.1 * (1 + AMPIEZZA))
    ax_r.set_xticks([0.5, 1.0, 1.5])              # evita sovrapposizione con il "10" dell'asse E_rec

    # Angolo in alto a destra: barra dei colori del kernel e parametri
    ax_cb = fig.add_axes([0.76, 0.86, 0.16, 0.018])
    cb = fig.colorbar(mesh, cax=ax_cb, orientation="horizontal")
    cb.set_label(r"$p(E_{rec}\,|\,E_{true})$ [MeV$^{-1}$]", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    testo = (rf"$\sigma/E = {100*A_STOC:.1f}\%/\sqrt{{E\,[\mathrm{{MeV}}]}}$" + "\n"
             rf"periodo vero = {PERIODO:g} MeV")
    fig.text(0.84, 0.76, testo, ha="center", va="center", fontsize=9)

    for out in ("risoluzione.png", "risoluzione.pdf"):
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Figura salvata in {out}")


if __name__ == "__main__":
    main()
