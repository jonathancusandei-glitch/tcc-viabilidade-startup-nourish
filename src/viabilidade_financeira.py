# -*- coding: utf-8 -*-
"""Modelagem Econômico-Financeira de 5 Anos (DRE, Curva de Maturação, VPL, TIR e Payback) — TCC Nourish."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def npv(rate: float, cashflows: list[float]) -> float:
    return sum(cf / ((1.0 + rate) ** t) for t, cf in enumerate(cashflows))


def irr(cashflows: list[float]) -> float:
    # Bisection root finder in [0, 3.0]
    lo, hi = -0.5, 3.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if npv(mid, cashflows) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    dre = pd.read_csv(os.path.join(DATA_DIR, "dre_5_anos.csv"))
    ll_row = dre[dre["Rubrica"] == "Lucro Liquido"].iloc[0, 1:].astype(float).values
    dep_row = dre[dre["Rubrica"] == "Depreciacao_Anual"].iloc[0, 1:].astype(float).values
    fc_oper = ll_row + dep_row

    investimento_inicial = 140000.0
    tma = 0.12
    cashflows = [-investimento_inicial] + list(fc_oper)
    vpl_calc = npv(tma, cashflows)
    tir_calc = irr(cashflows) * 100

    # Payback simples
    cum = np.cumsum(cashflows)
    payback = None
    for i in range(1, len(cum)):
        if cum[i] >= 0 and cum[i - 1] < 0:
            payback = (i - 1) + abs(cum[i - 1]) / fc_oper[i - 1]
            break

    # Curva mensal Ano 1 (Rampa 15 -> 70 refeicoes/dia, +5/mes, 26 dias, ticket R$ 32,50)
    meses = np.arange(1, 13)
    refeicoes_dia = 15 + 5 * (meses - 1)
    receita_m = refeicoes_dia * 26 * 32.50
    custo_var_m = receita_m * 0.45
    custo_fixo_m = np.full(12, 22850.0)
    custo_total_m = custo_var_m + custo_fixo_m

    print(f"Investimento Inicial: R$ {investimento_inicial:,.2f} (Fixo 55k + Pre-Op 25k + Giro 60k)")
    print(f"Fluxos 5 anos: {[round(x, 2) for x in cashflows]}")
    print(f"VPL (TMA 12%): R$ {vpl_calc:,.2f} | TIR: {tir_calc:.2f}% a.a. | Payback Simples: {payback:.2f} anos")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(meses, receita_m / 1000, marker="o", color="#10b981", linewidth=2.2, label="Receita Bruta (mil R$)")
    axes[0].plot(meses, custo_total_m / 1000, marker="s", color="#dc2626", linewidth=2.2, label="Custo Total (mil R$)")
    axes[0].axvline(8, color="#2563eb", linestyle="--", label="Ponto de Equilíbrio (Mês 8)")
    axes[0].set_title("Ano 1: Curva de Maturação & Break-Even", fontweight="bold")
    axes[0].set_xlabel("Mês de Operação")
    axes[0].set_ylabel("R$ Mil")
    axes[0].legend()

    anos = ["Ano 0", "Ano 1", "Ano 2", "Ano 3", "Ano 4", "Ano 5"]
    colors = ["#dc2626" if v < 0 else "#2563eb" for v in cashflows]
    axes[1].bar(anos, np.array(cashflows) / 1000, color=colors)
    axes[1].axhline(0, color="#0f172a", linewidth=1)
    axes[1].set_title(f"Fluxo de Caixa 5 Anos (TIR={tir_calc:.1f}% | Payback={payback:.2f}a)", fontweight="bold")
    axes[1].set_ylabel("R$ Mil")
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "viabilidade_maturacao_fluxo.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
