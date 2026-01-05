import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carga 
df = pd.read_csv("personas_2016_2025_todos_trimestres_limpio.csv")

# Limpieza mínima 
for c in ["ano4", "trimestre", "aglomerado", "estado"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["pondera"] = pd.to_numeric(df["pondera"], errors="coerce")

# Filtramos estados válidos y filas completas
df = df[df["estado"].isin([1, 2, 3])].dropna(subset=["ano4", "trimestre", "pondera", "estado"])

# Periodo ordenado 
df["PERIODO"] = df["ano4"].astype(int).astype(str) + "-T" + df["trimestre"].astype(int).astype(str)
orden = (df[["PERIODO","ano4","trimestre"]].drop_duplicates()
         .sort_values(["ano4","trimestre"]))["PERIODO"]
df["PERIODO"] = pd.Categorical(df["PERIODO"], categories=orden, ordered=True)

# Filtramos los aglomerados que usamos en el TP 
df = df[df["aglomerado"].isin([7, 9])]

# Tasa de actividad = (ocupados + desocupados) / poblacion total * 100
# PEA = estado 1 (ocupado) + estado 2 (desocupado)
# Total población por periodo y aglomerado
tot_pop = df.groupby(["PERIODO", "aglomerado"], observed=False)["pondera"].sum()

# PEA por periodo y aglomerado (ocupados + desocupados)
pea = df.loc[df["estado"].isin([1,2])].groupby(["PERIODO", "aglomerado"], observed=False)["pondera"].sum()

eps = 1e-12
tasa_activity = (pea / (tot_pop + eps) * 100).reset_index(name="tasa_actividad")

# nombres para etiquetas
nombres_aglo = {7: "Posadas", 9: "Comodoro Rivadavia–Rada Tilly"}
tasa_activity["aglomerado_str"] = tasa_activity["aglomerado"].map(nombres_aglo)

# helper para límites Y
def ylims_ajustados(s, margen=1.0):
    m, M = s.min(), s.max()
    return max(0, m - margen), M + margen

# Preparar series por aglomerado
df_com = tasa_activity[tasa_activity["aglomerado_str"] == "Comodoro Rivadavia–Rada Tilly"].sort_values("PERIODO")
df_pos = tasa_activity[tasa_activity["aglomerado_str"] == "Posadas"].sort_values("PERIODO")

# 1) Comodoro solo
ymin, ymax = ylims_ajustados(df_com["tasa_actividad"])
plt.figure(figsize=(9,4))
plt.plot(df_com["PERIODO"].astype(str), df_com["tasa_actividad"], marker="o", linewidth=2)
plt.ylim(ymin, ymax)
plt.title("Tasa de Actividad – Comodoro Rivadavia–Rada Tilly (2016–2025)", fontsize=12, weight="bold")
plt.ylabel("Tasa de Actividad (% población total)")
plt.xticks(rotation=90)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# 2) Posadas solo
ymin, ymax = ylims_ajustados(df_pos["tasa_actividad"])
plt.figure(figsize=(9,4))
plt.plot(df_pos["PERIODO"].astype(str), df_pos["tasa_actividad"], marker="o", linewidth=2)
plt.ylim(ymin, ymax)
plt.title("Tasa de Actividad – Posadas (2016–2025)", fontsize=12, weight="bold")
plt.ylabel("Tasa de Actividad (% población total)")
plt.xticks(rotation=90)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# 3) Comparativo
plt.figure(figsize=(10,5))
plt.plot(df_com["PERIODO"].astype(str), df_com["tasa_actividad"], marker="o", linewidth=2, label="Comodoro Rivadavia–Rada Tilly")
plt.plot(df_pos["PERIODO"].astype(str), df_pos["tasa_actividad"], marker="o", linewidth=2, label="Posadas")
ymin, ymax = ylims_ajustados(pd.concat([df_com["tasa_actividad"], df_pos["tasa_actividad"]]))
plt.ylim(ymin, ymax)
plt.title("Tasa de Actividad por Aglomerado (2016–2025)", fontsize=12, weight="bold")
plt.ylabel("Tasa de Actividad (% población total)")
plt.xticks(rotation=90)
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# (Opcional) imprimir tabla resumen si querés chequear números
print(tasa_activity.pivot(index="PERIODO", columns="aglomerado_str", values="tasa_actividad").round(2))
