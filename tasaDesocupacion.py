import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# carga 
df = pd.read_csv("personas_T2_2016_2025_posadas_comodoro_limpio_final.csv")

# Limpieza mínima 
for c in ["ano4", "trimestre", "aglomerado", "estado"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["pondera"] = pd.to_numeric(df["pondera"], errors="coerce")

# estados válidos: 1=ocupado, 2=inactivo, 3=desocupado
df = df[df["estado"].isin([1, 2, 3])].dropna(subset=["ano4","trimestre","pondera","estado"])

# Periodo ordenado 
df["PERIODO"] = df["ano4"].astype(int).astype(str) + "-T" + df["trimestre"].astype(int).astype(str)
orden = (df[["PERIODO","ano4","trimestre"]].drop_duplicates()
         .sort_values(["ano4","trimestre"]))["PERIODO"]
df["PERIODO"] = pd.Categorical(df["PERIODO"], categories=orden, ordered=True)

# Filtramos los dos aglomerados
df = df[df["aglomerado"].isin([7, 9])]


# TASA DE DESOCUPACIÓN (% de la PEA)
# desocupados (2) / (ocupados (1) + desocupados (2))
def tasa_desocup(gr):
    pea = gr.loc[gr["estado"].isin([1, 2]), "pondera"].sum()   # PEA = 1+2
    des = gr.loc[gr["estado"].eq(2), "pondera"].sum()          # Desocupados = 2
    return (des / pea * 100) if pea > 0 else np.nan

tasa_ag = (
    df.groupby(["PERIODO", "aglomerado"], observed=False)
      .apply(tasa_desocup)
      .reset_index(name="tasa_desocupacion")
)

# Mapa de nombres
nombres_aglo = {7: "Posadas", 9: "Comodoro Rivadavia–Rada Tilly"}
tasa_ag["aglomerado_str"] = tasa_ag["aglomerado"].map(nombres_aglo)


# Helpers para títulos y límites del eje Y
def ylims_ajustados(s, margen=0.5):
    m, M = s.min(), s.max()
    return max(0, m - margen), M + margen


# 1) Comodoro solo
df_com = tasa_ag[tasa_ag["aglomerado_str"] == "Comodoro Rivadavia–Rada Tilly"].sort_values("PERIODO")
ymin, ymax = ylims_ajustados(df_com["tasa_desocupacion"])

plt.figure(figsize=(9,4))
plt.plot(df_com["PERIODO"].astype(str), df_com["tasa_desocupacion"], marker="o", linewidth=2, label="Desocupación")
plt.ylim(ymin, ymax)
plt.title("Tasa de Desocupación – Comodoro Rivadavia–Rada Tilly (2016–2025)", fontsize=13, weight="bold")
plt.ylabel("Tasa de Desocupación (% de la PEA)")
plt.xticks(rotation=90); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# 2) Posadas solo
df_pos = tasa_ag[tasa_ag["aglomerado_str"] == "Posadas"].sort_values("PERIODO")
ymin, ymax = ylims_ajustados(df_pos["tasa_desocupacion"])

plt.figure(figsize=(9,4))
plt.plot(df_pos["PERIODO"].astype(str), df_pos["tasa_desocupacion"], marker="o", linewidth=2, color="darkorange", label="Desocupación")
plt.ylim(ymin, ymax)
plt.title("Tasa de Desocupación – Posadas (2016–2025)", fontsize=13, weight="bold")
plt.ylabel("Tasa de Desocupación (% de la PEA)")
plt.xticks(rotation=90); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()


# 3) Comparativo (dos líneas)
plt.figure(figsize=(10,5))
plt.plot(df_com["PERIODO"].astype(str), df_com["tasa_desocupacion"], marker="o", linewidth=2, label="Comodoro Rivadavia–Rada Tilly")
plt.plot(df_pos["PERIODO"].astype(str), df_pos["tasa_desocupacion"], marker="o", linewidth=2, color="darkorange", label="Posadas")

ymin, ymax = ylims_ajustados(pd.concat([df_com["tasa_desocupacion"], df_pos["tasa_desocupacion"]]))
plt.ylim(ymin, ymax)
plt.title("Tasa de Desocupación por Aglomerado (2016–2025)", fontsize=13, weight="bold")
plt.ylabel("Tasa de Desocupación %")
plt.xticks(rotation=90); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()