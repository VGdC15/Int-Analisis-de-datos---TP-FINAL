import pandas as pd
import numpy as np
from pathlib import Path

# ==========================
# CONFIG
# ==========================

CONFIG = {
    "year": "ANO4",
    "quarter": "TRIMESTRE",
    "agglomerate": "AGLOMERADO",
    "employment_status": "ESTADO",
    "age": "CH06",
    "hours": "PP3E_TOT",
    "income": "P21",
    "sex": "CH04",
    "education": "NIVEL_ED",
    "activity_branch": "PP04D_COD",
    "occupation_category": "CAT_OCUP",
    "establishment_size": "PP04G_COD",
    "city": "AGLOMERADO",
    "marital_status": "CH07",
    "household_id": "CODUSU",
    "dwelling_id": "NRO_HOGAR",
    "expansion_factor": "PONDERA",
    "formal_disc": "PP07H"
}

POSADAS = [7]
RADA_TILLY = [9]


# ==========================
# LOAD ALL TXT
# ==========================

def load_all_eph(folder="data"):
    path = Path(folder)
    files = sorted(path.glob("*.txt"))

    if not files:
        raise FileNotFoundError("No hay archivos .txt en /data")

    dfs = []
    for f in files:
        print(f"Leyendo {f}")
        dfs.append(pd.read_csv(f, sep=";", encoding="latin-1"))

    df = pd.concat(dfs, ignore_index=True)

    # Convertir columnas numéricas importantes
    numeric_cols = [
        "ANO4","TRIMESTRE","AGLOMERADO","ESTADO","CH06","PP3E_TOT",
        "P21","NIVEL_ED","PP04D_COD","CAT_OCUP","PP04G_COD","CH07","PP07H"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ==========================
# IPC + DESFLACTAR
# ==========================

def apply_ipc_deflation(df):
    """
    Une IPC trimestral y crea P21_real_2025.
    """

    ipc = pd.read_csv("ipc_trimestral.csv")  # ya existe en proyecto
    ipc["ano4"] = ipc["ano4"].astype(int)
    ipc["trimestre"] = ipc["trimestre"].astype(int)

    # unir por año + trimestre
    df = df.merge(
        ipc,
        left_on=["ANO4","TRIMESTRE"],
        right_on=["ano4","trimestre"],
        how="left"
    )

    # IPC de referencia: valores de 2025 T2
    ipc_ref = ipc[(ipc["ano4"] == 2025) & (ipc["trimestre"] == 2)]["IPC"].values[0]

    # ingreso real
    df["P21_real_2025"] = df["P21"] * (ipc_ref / df["IPC"])

    return df


# ==========================
# FILTROS
# ==========================

def filter_periods(df):
    c = CONFIG
    return df[
        (df[c["year"]].between(2017, 2025)) &
        (df[c["quarter"]] == 2) &
        (df[c["agglomerate"]].isin(POSADAS + RADA_TILLY))
    ].copy()


def select_occupied(df):
    c = CONFIG
    return df[
        (df[c["employment_status"]] == 1) &
        (df[c["age"]] >= 18)
    ].copy()


def remove_invalid_obs(df):
    c = CONFIG
    df = df[(df[c["age"]].between(18, 85))].copy()
    df = df[(df[c["hours"]] > 0) & (df[c["hours"]] <= 80)].copy()
    return df


# ==========================
# EDUCACIÓN
# ==========================

def map_education(df):
    c = CONFIG
    edu_map = {
        7: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        9: 0
    }
    df["educ_scale"] = df[c["education"]].map(edu_map).fillna(0).astype(int)
    return df


def create_variables(df):
    years_map = {
        0: 0,
        1: 3,
        2: 7,
        3: 9,
        4: 12,
        5: 15,
        6: 17
    }
    df["years_education"] = df["educ_scale"].map(years_map)

    df["formal"] = np.where(df["PP07H"] == 1, 1,
                            np.where(df["PP07H"] == 2, 0, np.nan))

    df["age2"] = df["CH06"] ** 2

    return df


# ==========================
# FALTANTES
# ==========================

def handle_missing(df):
    df["PP3E_TOT"] = df.groupby("CAT_OCUP")["PP3E_TOT"].transform(
        lambda x: x.fillna(x.median())
    )
    df["educ_scale"] = df["educ_scale"].fillna(0)
    df["PP04D_COD"] = df["PP04D_COD"].fillna("Desconocido")
    return df


# ==========================
# TRAIN / MISSING (con ingreso REAL)
# ==========================

def split_income_real(df):
    invalid = [-9, -8, -1, 0]

    df_train = df[
        (~df["P21_real_2025"].isna()) &
        (~df["P21"].isin(invalid)) &
        (df["P21"] > 0)
    ].copy()

    df_missing = df[
        (df["P21"].isin(invalid)) |
        (df["P21_real_2025"].isna())
    ].copy()

    return df_train, df_missing


# ==========================
# MAIN PIPELINE
# ==========================

def clean_eph():
    df = load_all_eph("data")

    df = filter_periods(df)
    df = select_occupied(df)
    df = remove_invalid_obs(df)

    # IPC + ingreso real
    df = apply_ipc_deflation(df)

    df = map_education(df)
    df = create_variables(df)
    df = handle_missing(df)

    df_train, df_missing = split_income_real(df)

    Path("processed").mkdir(exist_ok=True)

    df_train.to_csv("processed/eph_train_ingreso_real.csv", index=False)
    df_missing.to_csv("processed/eph_missing_ingreso_real.csv", index=False)

    print("Limpieza completa con ingreso real.")
    print("Filas TRAIN:", len(df_train))
    print("Filas MISSING:", len(df_missing))


if __name__ == "__main__":
    clean_eph()
