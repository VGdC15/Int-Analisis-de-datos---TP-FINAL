# ============================================================
# LIMPIEZA EPH PERSONAS (2016–2025, TODOS LOS TRIMESTRES)
# Reglas del profesor:
#   - Edad >= 18
#   - Solo ocupados (1) y desocupados (2)
#   - Sin límite de edad superior
#   - Mantener TODOS los trimestres
#   - Tratar No respuesta
#   - Eliminar Duplicados
#   - Eliminar Outliers (ingresos por cuantil anual)
#
# Aglomerados: Posadas (7) y Comodoro Rivadavia–Rada Tilly (9)
# ============================================================

import pandas as pd
from pathlib import Path

# ---------------------- CONFIG -----------------------
INPUT_DIR  = Path("data")   # Carpeta con todos los TXT
OUTPUT_PATH = "personas_2016_2025_todos_trimestres_limpio_sin_outliers.csv"

CLAVE = ["CODUSU", "NRO_HOGAR", "COMPONENTE", "ANO4", "TRIMESTRE", "AGLOMERADO"]

COLS_KEEP = [
    "CODUSU","NRO_HOGAR","COMPONENTE",
    "ANO4","TRIMESTRE","AGLOMERADO",
    "H15","PONDERA",
    "ESTADO","CAT_OCUP","CH04","CH06",
    "PP3E_TOT","PP3F_TOT","P47T"
]

INT_COLS = ["ANO4","TRIMESTRE","AGLOMERADO","NRO_HOGAR","COMPONENTE",
            "H15","CH06","ESTADO","CAT_OCUP"]


# ============================================================
# CARGA DE ARCHIVOS
# ============================================================

def cargar_multiples_txt(input_dir: Path) -> pd.DataFrame:
    frames = []
    files = sorted(input_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No se encontraron TXT en {input_dir.resolve()}")

    for f in files:
        try:
            df = pd.read_csv(f, sep=";", dtype=str)
            cols = [c for c in COLS_KEEP if c in df.columns]
            if not cols:
                continue

            df = df[cols].copy()
            df["CODUSU"] = df["CODUSU"].astype(str).str.strip()
            df["__archivo_origen"] = f.name
            frames.append(df)
            print(f"   + leído: {f.name} | filas: {len(df)}")
        except Exception as e:
            print(f"   ! error leyendo {f.name}: {e}")

    return pd.concat(frames, ignore_index=True)


# ============================================================
# TIPADO
# ============================================================

def tipar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in INT_COLS:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    if "PONDERA" in df:
        df["PONDERA"] = pd.to_numeric(df["PONDERA"], errors="coerce")

    for c in ["PP3E_TOT","PP3F_TOT","P47T"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ============================================================
# OUTLIERS DE INGRESO
# ============================================================

def eliminar_outliers_ingresos_por_anio(df: pd.DataFrame, col="P47T", q=0.995) -> pd.DataFrame:
    df = df.copy()
    if col not in df.columns:
        return df

    def _clean(gr):
        valid = gr[col].dropna()
        if valid.empty:
            return gr
        lim = valid.quantile(q)
        gr.loc[gr[col] > lim, col] = pd.NA
        return gr

    return df.groupby("ANO4", group_keys=False).apply(_clean)


# ============================================================
# FILTROS (según profe)
# ============================================================

def filtrar_universo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # AÑOS DEL TP
    df = df[df["ANO4"].between(2016, 2025)]

    # Aglomerados del TP
    df = df[df["AGLOMERADO"].isin([7, 9])]

    # Entrevista individual realizada
    df = df[df["H15"] == 1]

    # Edad >= 18 (con control de edades imposibles)
    df.loc[(df["CH06"] < 0) | (df["CH06"] > 110), "CH06"] = pd.NA
    df = df[df["CH06"].notna()]
    df = df[df["CH06"] >= 18]

    # ESTADO: solo ocupados (1) y desocupados (2)
    df.loc[df["ESTADO"].isin([9,99,999]), "ESTADO"] = pd.NA
    df = df[df["ESTADO"].isin([1,2])]

    return df


# ============================================================
# SANIDAD BÁSICA
# ============================================================

def sanidad_basica(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in ["PP3E_TOT","PP3F_TOT"]:
        if c in df:
            df.loc[(df[c] < 0) | (df[c] > 168), c] = pd.NA

    if "CAT_OCUP" in df:
        df.loc[df["CAT_OCUP"].isin([9,99,999]), "CAT_OCUP"] = pd.NA

    return df


# ============================================================
# DUPLICADOS
# ============================================================

def elegir_mejor(gr: pd.DataFrame) -> pd.DataFrame:
    vars_info = [c for c in
                 ["ESTADO","CAT_OCUP","CH04","CH06","PP3E_TOT","PP3F_TOT","P47T","PONDERA","H15"]
                 if c in gr.columns]

    s1 = (gr["H15"] == 1).astype(int)
    s2 = gr[vars_info].notna().sum(axis=1)
    s3 = gr["PONDERA"].notna().astype(int)

    score = s1*1000 + s2*10 + s3
    return gr.loc[[score.idxmax()]]


def resolver_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not df.duplicated(subset=CLAVE).any():
        return df.reset_index(drop=True)

    return (
        df.groupby(CLAVE, group_keys=False)
          .apply(lambda g: g if len(g)==1 else elegir_mejor(g))
          .reset_index(drop=True)
    )


# ============================================================
# NORMALIZAR NOMBRES
# ============================================================

def normalizar_nombres(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = df2.columns.str.lower()
    return df2


# ============================================================
# MAIN
# ============================================================

def main():
    print("1) Cargando TXT…")
    df = cargar_multiples_txt(INPUT_DIR)
    print(f"   Filas leídas: {len(df)}")

    print("2) Tipando columnas…")
    df = tipar_columnas(df)

    print("3) Filtrando universo…")
    df = filtrar_universo(df)
    print(f"   Filas tras filtros: {len(df)}")

    print("4) Sanidad básica…")
    df = sanidad_basica(df)

    print("5) Eliminando outliers de ingresos…")
    df = eliminar_outliers_ingresos_por_anio(df, "P47T", q=0.995)

    print("6) Resolviendo duplicados…")
    df = resolver_duplicados(df)

    print("7) Normalizando nombres…")
    df = normalizar_nombres(df)

    print("8) Guardando archivo final…")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Listo: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
