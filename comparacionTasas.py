import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) CARGA Y LIMPIEZA MÍNIMA
df = pd.read_csv("personas_T2_2016_2025_posadas_comodoro_limpio_final.csv")

# Tipos necesarios
for c in ["ano4", "trimestre", "aglomerado", "estado"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["pondera"] = pd.to_numeric(df["pondera"], errors="coerce")

# estados válidos: 1=ocupado, 2=desocupado, 3=inactivo
df = df[df["estado"].isin([1, 2, 3])].dropna(subset=["ano4", "trimestre", "pondera", "estado"])

# 2) VARIABLE PERIODO
df["PERIODO"] = df["ano4"].astype(int).astype(str) + "-T" + df["trimestre"].astype(int).astype(str)

orden = (
    df[["PERIODO", "ano4", "trimestre"]]
    .drop_duplicates()
    .sort_values(["ano4", "trimestre"])
)["PERIODO"]

df["PERIODO"] = pd.Categorical(df["PERIODO"], categories=orden, ordered=True)
df = df[df["aglomerado"].isin([7, 9])]

# Nombres 
nombres_aglo = {7: "Posadas", 9: "Comodoro Rivadavia–Rada Tilly"}
df["aglomerado_str"] = df["aglomerado"].map(nombres_aglo)

# 3) FUNCIONES PARA CALCULAR LAS TRES TASAS
def tasa_actividad(gr):
    pea = gr.loc[gr["estado"].isin([1,2]), "pondera"].sum()
    pob = gr["pondera"].sum()
    return (pea / pob * 100) if pob > 0 else np.nan

def tasa_empleo(gr):
    emp = gr.loc[gr["estado"] == 1, "pondera"].sum()
    pob = gr["pondera"].sum()
    return (emp / pob * 100) if pob > 0 else np.nan

def tasa_desocup(gr):
    pea = gr.loc[gr["estado"].isin([1,2]), "pondera"].sum()
    des = gr.loc[gr["estado"] == 2, "pondera"].sum()
    return (des / pea * 100) if pea > 0 else np.nan

# Calculamos tasas
tasas = df.groupby(["PERIODO", "aglomerado_str"], observed=False).apply(
    lambda g: pd.Series({
        "actividad": tasa_actividad(g),
        "empleo": tasa_empleo(g),
        "desocupacion": tasa_desocup(g)
    })
).reset_index()

# 4) HELPERS PARA LÍMITES
def ylims(s, margen=1):
    m, M = s.min(), s.max()
    return max(0, m - margen), M + margen

# 5) GRAFICO 1 – POSADAS
df_pos = tasas[tasas["aglomerado_str"] == "Posadas"].sort_values("PERIODO")
ymin, ymax = ylims(pd.concat([df_pos["actividad"], df_pos["empleo"], df_pos["desocupacion"]]))

plt.figure(figsize=(11,5))
plt.plot(df_pos["PERIODO"], df_pos["actividad"], marker="o", linewidth=2, label="Actividad")
plt.plot(df_pos["PERIODO"], df_pos["empleo"], marker="o", linewidth=2, label="Empleo")
plt.plot(df_pos["PERIODO"], df_pos["desocupacion"], marker="o", linewidth=2, label="Desocupación")

plt.ylim(ymin, ymax)
plt.title("Tasas Laborales – Posadas (2016–2025, T2)", fontsize=14, weight="bold")
plt.ylabel("Tasa (%)")
plt.xticks(rotation=90)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# 6) GRAFICO 2 – COMODORO RIVADAVIA–RADA TILLY
df_com = tasas[tasas["aglomerado_str"] == "Comodoro Rivadavia–Rada Tilly"].sort_values("PERIODO")
ymin, ymax = ylims(pd.concat([df_com["actividad"], df_com["empleo"], df_com["desocupacion"]]))

plt.figure(figsize=(11,5))
plt.plot(df_com["PERIODO"], df_com["actividad"], marker="o", linewidth=2, label="Actividad")
plt.plot(df_com["PERIODO"], df_com["empleo"], marker="o", linewidth=2, label="Empleo")   
plt.plot(df_com["PERIODO"], df_com["desocupacion"], marker="o", linewidth=2, label="Desocupación")

plt.ylim(ymin, ymax)
plt.title("Tasas Laborales – Comodoro Rivadavia–Rada Tilly (2016–2025, T2)", fontsize=14, weight="bold")
plt.ylabel("Tasa (%)")
plt.xticks(rotation=90)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
