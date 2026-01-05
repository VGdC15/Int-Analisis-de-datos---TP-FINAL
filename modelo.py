import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


# ==========================
# CARGAR DATOS LIMPIOS
# ==========================

def load_clean_data():
    base = Path("processed")
    df_train = pd.read_csv(base / "eph_train_ingreso_real.csv")
    df_missing = pd.read_csv(base / "eph_missing_ingreso_real.csv")
    return df_train, df_missing


# ==========================
# ARMAR FEATURES (RENOMBRANDO COLUMNAS)
# ==========================

def build_feature_sets(df_train):

    target = "P21_real_2025"

    # MAPEO COMPLETO
    rename_map = {
        "CH06": "edad",
        "age2": "edad_cuadrado",
        "years_education": "anios_educacion",
        "formal": "formalidad",
        "PP3E_TOT": "horas_trabajadas",
        "CH04": "sexo",
        "NIVEL_ED": "nivel_educativo",
        "PP04D_COD": "rama_actividad",
        "CAT_OCUP": "categoria_ocupacional",
        "PP04G_COD": "tamano_establecimiento",
        "CH07": "estado_civil"
    }

    # RENOMBRAR SOLO LAS QUE EXISTEN
    cols_presentes = {k: v for k, v in rename_map.items() if k in df_train.columns}
    df_train = df_train.rename(columns=cols_presentes)

    # ARMAMOS FEATURE_COLS SOLO CON LAS COLUMNAS QUE EXISTEN
    feature_cols = list(cols_presentes.values())

    # EL TARGET SI EXISTE
    if target not in df_train.columns:
        raise KeyError("No existe la columna 'P21_real_2025' en df_train.")

    # FILTRAR FILAS COMPLETAS
    df_model = df_train[feature_cols + [target]].dropna()

    X = df_model[feature_cols]
    y = df_model[target]

    # SEPARACIÓN ENTRE NUMÉRICAS Y CATEGÓRICAS
    numeric_candidates = [
        "edad", "edad_cuadrado", "anios_educacion",
        "formalidad", "horas_trabajadas"
    ]

    categorical_candidates = [
        "sexo", "nivel_educativo", "rama_actividad",
        "categoria_ocupacional", "tamano_establecimiento", "estado_civil"
    ]

    numeric = [c for c in numeric_candidates if c in feature_cols]
    categorical = [c for c in categorical_candidates if c in feature_cols]

    return X, y, numeric, categorical, feature_cols


# ==========================
# PIPELINE
# ==========================

def build_pipeline(numeric, categorical):

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
        ],
        remainder="drop"
    )

    model = DecisionTreeRegressor(
        max_depth=6,
        min_samples_leaf=60,
        random_state=42
    )

    return Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])


# ==========================
# ENTRENAR + MÉTRICAS
# ==========================

def train_and_evaluate(X, y, model, name):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(f"\n=== METRICAS {name.upper()} ===")
    print(f"MAE  : {mean_absolute_error(y_test, pred):,.2f}")
    print(f"RMSE : {mean_squared_error(y_test, pred)**0.5:,.2f}")
    print(f"R²   : {r2_score(y_test, pred):.3f}")

    return model


# ==========================
# ARBOL CON NOMBRES HUMANOS
# ==========================

def plot_tree_graph(model, features, name):

    tree = model.named_steps["model"]
    prep = model.named_steps["prep"]

    raw = prep.get_feature_names_out(features)

    pretty = []
    for n in raw:
        n = n.replace("num__", "").replace("cat__", "")
        pretty.append(n.split("_")[0])  # elimina sufijos de categorías

    plt.figure(figsize=(22, 10))
    plot_tree(tree, feature_names=pretty, filled=False, max_depth=3, fontsize=6)

    out = Path("processed") / f"arbol_{name}.png"
    plt.savefig(out, dpi=200)
    plt.close()

    print(f"Árbol guardado → {out}")


# ==========================
# MODELO POR AGLOMERADO
# ==========================

def model_for_city(df_train, df_missing, city_name):

    print(f"\n=== MODELO {city_name.upper()} ===")

    if df_train.empty:
        print(f"[AVISO] No hay datos para {city_name}.")
        return None, None, None

    X, y, numeric, categorical, feature_cols = build_feature_sets(df_train)

    print(f"Columnas usadas ({city_name}): {feature_cols}")
    print(f"Filas para entrenar: {len(X)}")

    pipe = build_pipeline(numeric, categorical)

    pipe = train_and_evaluate(X, y, pipe, city_name)

    plot_tree_graph(pipe, feature_cols, city_name)

    # ENTRENAR CON TODO PARA IMPUTAR
    pipe.fit(X, y)

    # IMPUTAR FALTANTES
    df_missing_imp = df_missing.copy()
    if not df_missing.empty:

        # RENOMBRAR COLUMNAS EN df_missing
        rename_map = {
            "CH06": "edad",
            "age2": "edad_cuadrado",
            "years_education": "anios_educacion",
            "formal": "formalidad",
            "PP3E_TOT": "horas_trabajadas",
            "CH04": "sexo",
            "NIVEL_ED": "nivel_educativo",
            "PP04D_COD": "rama_actividad",
            "CAT_OCUP": "categoria_ocupacional",
            "PP04G_COD": "tamano_establecimiento",
            "CH07": "estado_civil"
        }

        rename_present = {k: v for k, v in rename_map.items() if k in df_missing.columns}
        df_missing_ren = df_missing.rename(columns=rename_present)

        X_miss = df_missing_ren[feature_cols].copy().fillna(0)
        df_missing_imp["P21_real_2025_imputado"] = pipe.predict(X_miss)

    # GUARDAR TRAIN PRED
    df_train_pred = df_train.copy()
    df_train_pred["P21_real_2025_predicho"] = np.nan
    df_train_pred.loc[X.index, "P21_real_2025_predicho"] = pipe.predict(X)

    out = Path("processed")
    out.mkdir(exist_ok=True)

    df_train_pred.to_csv(out / f"train_pred_{city_name}.csv", index=False)
    df_missing_imp.to_csv(out / f"missing_imputado_{city_name}.csv", index=False)

    print(f"\nGenerados para {city_name}:")
    print(f"→ train_pred_{city_name}.csv")
    print(f"→ missing_imputado_{city_name}.csv")

    return pipe, df_train_pred, df_missing_imp


# ==========================
# MAIN
# ==========================

if __name__ == "__main__":

    df_train, df_missing = load_clean_data()

    df_train_pos = df_train[df_train["AGLOMERADO"] == 7]
    df_missing_pos = df_missing[df_missing["AGLOMERADO"] == 7]

    df_train_rt = df_train[df_train["AGLOMERADO"] == 9]
    df_missing_rt = df_missing[df_missing["AGLOMERADO"] == 9]

    model_pos, pred_pos, missing_pos = model_for_city(df_train_pos, df_missing_pos, "posadas")
    model_rt, pred_rt, missing_rt = model_for_city(df_train_rt, df_missing_rt, "rada_tilly")
