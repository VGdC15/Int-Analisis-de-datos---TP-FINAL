import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carga 
df = pd.read_csv("personas_2016_2025_todos_trimestres_limpio.csv")

# Limpieza mínima 
for c in ["ano4", "trimestre", "aglomerado", "estado"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["pondera"] = pd.to_numeric(df["pondera"], errors="coerce")

df = df[df["estado"].isin([1,2,3])].dropna(subset=["ano4","trimestre","pondera","estado"])

# Periodo ordenado 
df["PERIODO"] = df["ano4"].astype(int).astype(str) + "-T" + df["trimestre"].astype(int).astype(str)
orden = (df[["PERIODO","ano4","trimestre"]].drop_duplicates()
         .sort_values(["ano4","trimestre"]))["PERIODO"]
df["PERIODO"] = pd.Categorical(df["PERIODO"], categories=orden, ordered=True)

# Filtramos los aglomerados del TP 
df = df[df["aglomerado"].isin([7,9])]


# Tasa de empleo = ocupados / población total * 100
# ocupados => estado == 1
eps = 1e-12

ocup_ag = (df["pondera"] * df["estado"].eq(1).astype(int)).groupby(
    [df["PERIODO"], df["aglomerado"]], observed=False
).sum()
tot_pop = df.groupby([df["PERIODO"], df["aglomerado"]], observed=False)["pondera"].sum()

tasa_emp = (ocup_ag / (tot_pop + eps) * 100).reset_index(name="tasa_empleo")

# nombres para etiquetas
nombres_aglo = {7: "Posadas", 9: "Comodoro Rivadavia–Rada Tilly"}
tasa_emp["aglomerado_str"] = tasa_emp["aglomerado"].map(nombres_aglo)

# Helpers para límites Y
def ylims_ajustados(s, margen=1.0):
    m, M = s.min(), s.max()
    return max(0, m - margen), M + margen

# series por aglomerado
df_com = tasa_emp[tasa_emp["aglomerado_str"] == "Comodoro Rivadavia–Rada Tilly"].sort_values("PERIODO")
df_pos = tasa_emp[tasa_emp["aglomerado_str"] == "Posadas"].sort_values("PERIODO")

# 1) Comodoro solo
ymin, ymax = ylims_ajustados(df_com["tasa_empleo"])
plt.figure(figsize=(9,4))
plt.plot(df_com["PERIODO"].astype(str), df_com["tasa_empleo"], marker="o", linewidth=2)
plt.ylim(ymin, ymax)
plt.title("Tasa de Empleo – Comodoro Rivadavia–Rada Tilly (2016–2025)", fontsize=12, weight="bold")
plt.ylabel("Tasa de Empleo (% población total)")
plt.xticks(rotation=90)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
plt.close()

# 2) Posadas solo
ymin, ymax = ylims_ajustados(df_pos["tasa_empleo"])
plt.figure(figsize=(9,4))
plt.plot(df_pos["PERIODO"].astype(str), df_pos["tasa_empleo"], marker="o", linewidth=2)
plt.ylim(ymin, ymax)
plt.title("Tasa de Empleo – Posadas (2016–2025)", fontsize=12, weight="bold")
plt.ylabel("Tasa de Empleo (% población total)")
plt.xticks(rotation=90)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
plt.close()


# 3) Comparativo
plt.figure(figsize=(10,5))
plt.plot(df_com["PERIODO"].astype(str), df_com["tasa_empleo"], marker="o", linewidth=2, label="Comodoro Rivadavia–Rada Tilly")
plt.plot(df_pos["PERIODO"].astype(str), df_pos["tasa_empleo"], marker="o", linewidth=2, label="Posadas")
ymin, ymax = ylims_ajustados(pd.concat([df_com["tasa_empleo"], df_pos["tasa_empleo"]]))
plt.ylim(ymin, ymax)
plt.title("Tasa de Empleo por Aglomerado (2016–2025)", fontsize=12, weight="bold")
plt.ylabel("Tasa de Empleo (% población total)")
plt.xticks(rotation=90)
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
plt.close()

# (Opcional) imprimir tabla resumen para chequear valores
print(tasa_emp.pivot(index="PERIODO", columns="aglomerado_str", values="tasa_empleo").round(2))
