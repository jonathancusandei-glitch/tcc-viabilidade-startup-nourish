# -*- coding: utf-8 -*-
"""Análise Estatística da Pesquisa Quantitativa de Mercado (N = 156) — TCC UFJF (Nourish Ltda.)."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "pesquisa_mercado_anonimizada.csv")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def clean_likert(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    return int(s[0]) if s and s[0].isdigit() else np.nan


def main():
    df = pd.read_csv(CSV_PATH)
    df.columns = [
        "id", "concorda", "reside_gv", "faixa_etaria", "sexo_nascimento", "genero",
        "renda_familiar", "restricoes", "frequencia_consumo", "canais_compra",
        "likert_dificuldade", "likert_preco_premium", "likert_info_nutricional",
        "likert_impulso_insta", "likert_intencao_compra",
        "imp_variedade", "imp_atendimento_zap", "imp_localizacao_centro",
        "imp_recomendacao_prof", "imp_foco_exclusivo",
        "comunicacao_estilo", "anuncios_foco", "identidade_visual",
    ]

    likert_cols = [
        "likert_dificuldade", "likert_preco_premium", "likert_info_nutricional",
        "likert_impulso_insta", "likert_intencao_compra",
        "imp_variedade", "imp_atendimento_zap", "imp_localizacao_centro",
        "imp_recomendacao_prof", "imp_foco_exclusivo",
    ]
    for c in likert_cols:
        df[c] = df[c].apply(clean_likert)

    # Identifica subgrupo com alguma restrição/dieta (excluindo "Nenhuma")
    has_rest = ~df["restricoes"].fillna("Nenhuma").str.strip().eq("Nenhuma")
    escassez_rest = (df.loc[has_rest, "likert_dificuldade"] >= 5).mean() * 100
    escassez_sem = (df.loc[~has_rest, "likert_dificuldade"] >= 5).mean() * 100

    print(f"Amostra N={len(df)} | Publico com restricao/dieta: {has_rest.sum()} ({has_rest.mean()*100:.1f}%)")
    print(f"Escassez percebida (Likert >= 5): Restritivos={escassez_rest:.1f}% vs Nao restritivos={escassez_sem:.1f}%")
    print(f"Clareza nutricional Media={df['likert_info_nutricional'].mean():.2f} | Compra impulso Insta={df['likert_impulso_insta'].mean():.2f}")

    # Gráficos
    plt.style.use("seaborn-v0_8-whitegrid")
    fatores = {
        "Variedade cardápio": df["imp_variedade"].mean(),
        "Atendimento WhatsApp": df["imp_atendimento_zap"].mean(),
        "Indicação médica/nutri": df["imp_recomendacao_prof"].mean(),
        "Localização Centro": df["imp_localizacao_centro"].mean(),
        "Foco exclusivo inclusivo": df["imp_foco_exclusivo"].mean(),
    }
    s_fat = pd.Series(fatores).sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    s_fat.plot(kind="barh", color="#2563eb", ax=axes[0])
    axes[0].set_xlim(0, 7)
    axes[0].set_title("Fatores Decisórios de Compra (Likert 1–7)", fontweight="bold")
    axes[0].set_xlabel("Média na Amostra (N=156)")

    axes[1].bar(["Com Restrição/Dieta", "Sem Restrição"], [escassez_rest, escassez_sem], color=["#dc2626", "#94a3b8"])
    axes[1].set_ylabel("% Concordância Dificuldade (Likert ≥ 5)")
    axes[1].set_title("Lacuna de Oferta Segura (~6x maior nos restritivos)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "pesquisa_mercado_insights.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
