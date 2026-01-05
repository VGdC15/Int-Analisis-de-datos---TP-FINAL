# ============================================================
# LIMPIEZA EPH PERSONAS (2016–2025, TODOS LOS TRIMESTRES)
# Aglomerados: Posadas (7) y Comodoro Rivadavia–Rada Tilly (9)
# Criterios según profe:
#   - Edad >= 18
#   - Solo ocupados y desocupados (sacamos inactivos)
#   - Sin límite de edad superior (no cortamos 65/64)
#   - Mantener todos los trimestres
#   - Tratar duplicados
#   - Tratar no respuesta
# ============================================================

import pandas as pd
from pathlib import Path

# ---------------------- Configuración -----------------------
INPUT_DIR  = Path("data")  # carpeta con los TXT de usu_individual
OUTPUT_PATH = "personas_2016_2025_todos_trimestres_limpio.csv"

# Clave de unicidad por persona/tiempo/aglomerado
CLAVE = ["CODUSU", "NRO_HOGAR", "COMPONENTE", "ANO4", "TRIMESTRE", "AGLOMERADO"]

# Columnas que se conservan
COLS_KEEP = [
    "CODUSU","NRO_HOGAR","COMPONENTE",
    "ANO4","TRIMESTRE","AGLOMERADO",
    "H15","PONDERA",
    "ESTADO","CAT_OCUP","CH04","CH06",
    "PP3E_TOT","PP3F_TOT","P47T"
]

# Columnas a tipar como enteros con nulos
INT_COLS = ["ANO4","TRIMESTRE","AGLOMERADO","NRO_HOGAR","COMPONENTE","H15","CH06","ESTADO","CAT_OCUP"]


# ---------------------- Funciones base ----------------------
def cargar_multiples_txt(input_dir: Path) -> pd.DataFrame:
    """
    Lee todos los TXT en INPUT_DIR (sep=';'), conserva columnas esperadas
    y concatena en un único DataFrame. Ignora archivos sin columnas clave.
    """
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
            if "CODUSU" in df.columns:
                df["CODUSU"] = df["CODUSU"].astype(str).str.strip()
            df["__archivo_origen"] = f.name  # tracking opcional
            frames.append(df)
            print(f"   + leído: {f.name} | filas: {len(df)}")
        except Exception as e:
            print(f"   ! error leyendo {f.name}: {e}")
    if not frames:
        raise ValueError("No se pudo leer ningún archivo con las columnas esperadas.")
    return pd.concat(frames, ignore_index=True)


def tipar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte a numérico lo necesario; PONDERA/horas/ingreso como float."""
    df = df.copy()
    for c in INT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    if "PONDERA" in df.columns:
        df["PONDERA"] = pd.to_numeric(df["PONDERA"], errors="coerce")
    for c in ["PP3E_TOT","PP3F_TOT","P47T"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def filtrar_universo(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Años 2016–2025
    - Todos los trimestres (NO filtramos TRIMESTRE)
    - Aglomerados 7 y 9
    - Entrevista individual realizada (H15 == 1)
    - Edad >= 18 (sin tope superior)
    - Solo ESTADO 1 y 2 (ocupados y desocupados)
    """
    df = df.copy()

    # años del TP
    if "ANO4" in df.columns:
        df = df[df["ANO4"].between(2016, 2025)]

    # aglomerados del TP
    if "AGLOMERADO" in df.columns:
        df = df[df["AGLOMERADO"].isin([7, 9])]

    # entrevista individual realizada
    if "H15" in df.columns:
        df = df[df["H15"] == 1]

    # limpieza básica de edad antes de corte
    if "CH06" in df.columns:
        # seguimos evitando edades imposibles, pero sin tope analítico (no cortamos en 64/65)
        df.loc[(df["CH06"] < 0) | (df["CH06"] > 110), "CH06"] = pd.NA
        df = df[df["CH06"].notna()]
        df = df[df["CH06"] >= 18]

    # ESTADO: sólo ocupados y desocupados (sacamos inactivos y Ns/Nr)
    if "ESTADO" in df.columns:
        # primero tratamos no respuesta
        df.loc[df["ESTADO"].isin([9, 99, 999]), "ESTADO"] = pd.NA
        df = df[df["ESTADO"].isin([1, 2])]

    return df


def sanidad_basica(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanidad mínima:
      - Horas en rango razonable [0,168]
      - Ns/Nr en algunas variables → NaN
    """
    df = df.copy()

    # Horas semanales razonables
    for c in ["PP3E_TOT","PP3F_TOT"]:
        if c in df.columns:
            df.loc[(df[c] < 0) | (df[c] > 168), c] = pd.NA

    # Ns/Nr en CAT_OCUP (por si se usa después)
    if "CAT_OCUP" in df.columns:
        df.loc[df["CAT_OCUP"].isin([9, 99, 999]), "CAT_OCUP"] = pd.NA

    return df


def diagnostico_duplicados(df: pd.DataFrame) -> dict:
    """Resumen de duplicados por CLAVE (totales, exactos, conflictivos)."""
    mask_clave = df.duplicated(subset=CLAVE, keep=False)
    mask_exact = df.duplicated(keep=False)
    dups_total = int(mask_clave.sum())
    exactos = int((mask_clave & mask_exact).sum())
    conflictivos = int(dups_total - exactos)
    grupos_conf = (
        df.loc[mask_clave, CLAVE]
          .value_counts()
          .reset_index(name="n")
          .query("n > 1")
          .shape[0]
    )
    return {
        "filas": len(df),
        "dups_total_filas": dups_total,
        "dups_exactos_filas": exactos,
        "dups_conflictivos_filas": conflictivos,
        "grupos_clave_con_multiples_filas": int(grupos_conf),
    }


def elegir_mejor(gr: pd.DataFrame) -> pd.DataFrame:
    """
    Selección de la mejor fila por CLAVE:
      1) H15 == 1
      2) Más campos informativos no nulos
      3) PONDERA no nulo
      4) Si empata, primera aparición
    """
    gr = gr.copy()
    vars_info = [c for c in ["ESTADO","CAT_OCUP","CH04","CH06","PP3E_TOT","PP3F_TOT","P47T","PONDERA","H15"] if c in gr.columns]
    s1 = (gr["H15"] == 1).astype(int) if "H15" in gr.columns else 0
    s2 = gr[vars_info].notna().sum(axis=1) if vars_info else 0
    s3 = gr["PONDERA"].notna().astype(int) if "PONDERA" in gr.columns else 0
    score = s1*1000 + s2*10 + s3
    return gr.loc[[score.idxmax()]]


def resolver_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """Compacta exactos y resuelve conflictivos aplicando elegir_mejor()."""
    df = df.copy()
    if not df.duplicated(subset=CLAVE).any():
        return df.reset_index(drop=True)
    return (
        df.groupby(CLAVE, group_keys=False)
          .apply(lambda g: g if len(g)==1 else elegir_mejor(g))
          .reset_index(drop=True)
    )


def normalizar_nombres(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte nombres de columnas a minúsculas al final del pipeline."""
    df2 = df.copy()
    df2.columns = df2.columns.str.lower()
    return df2


# ---------------------- Proceso principal -------------------
def main():
    print("1) Cargando múltiples archivos TXT…")
    df = cargar_multiples_txt(INPUT_DIR)
    print(f"   TOTAL filas leídas: {len(df)}")

    print("2) Tipando columnas…")
    df = tipar_columnas(df)

    print("3) Filtro universo (2016–2025, todos los trimestres, 18+, ESTADO 1/2)…")
    df = filtrar_universo(df)
    print(f"   Filas tras filtro: {len(df)}")

    print("4) Sanidad básica (horas, Ns/Nr)…")
    df = sanidad_basica(df)

    print("5) Diagnóstico duplicados (ANTES)…")
    diag_b = diagnostico_duplicados(df)
    for k,v in diag_b.items():
        print(f"   {k}: {v}")

    print("6) Resolviendo duplicados…")
    df = resolver_duplicados(df)

    print("7) Diagnóstico duplicados (DESPUÉS)…")
    diag_a = diagnostico_duplicados(df)
    for k,v in diag_a.items():
        print(f"   {k}: {v}")

    print("8) Normalizando nombres de columnas…")
    df = normalizar_nombres(df)

    print("9) Guardando CSV final…")
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Listo: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
