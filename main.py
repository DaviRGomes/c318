import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

def load_indicator_prices(path):
    df = pd.read_csv(path, sep=";", dtype=str)
    df["months"] = pd.to_datetime(df["months"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["months"])
    s = df["ICO composite indicator"].astype(str)
    mant = s.str.split("E", n=1, expand=True)[0].str.replace(",", ".", regex=False).str.replace(" ", "", regex=False)
    vals = pd.to_numeric(mant, errors="coerce") * 100.0
    df["ICO composite indicator"] = vals
    df = df.set_index("months")
    y_annual = df.groupby(df.index.year)["ICO composite indicator"].mean().rename("Y_anual").to_frame()
    y_annual.index.name = "Ano"
    return y_annual

def load_annual_wide(path):
    raw = pd.read_csv(path, sep=";", header=None, dtype=str)
    header = list(raw.iloc[1])
    data = raw.iloc[2:].copy()
    data.columns = header
    first_col = header[0]
    data = data.rename(columns={first_col: "country"})
    year_cols = [c for c in data.columns if c.isdigit()]
    for c in year_cols:
        s = data[c].astype(str).str.replace(",", ".", regex=False).str.replace(" ", "", regex=False)
        data[c] = pd.to_numeric(s, errors="coerce")
    data = data.set_index("country")
    data.index = data.index.str.strip()
    df_t = data[year_cols].transpose()
    df_t.index = df_t.index.astype(int)
    df_t.index.name = "Ano"
    return df_t

def impute_and_melt(path, value_name):
    wide = load_annual_wide(path)
    wide = impute_by_column(wide)
    df_long = wide.reset_index().melt(id_vars="Ano", var_name="Pais", value_name=value_name)
    df_long = df_long.groupby(["Ano", "Pais"], as_index=False).mean()
    if value_name == "Preco_Produtor":
        df_long[value_name] = pd.to_numeric(df_long[value_name], errors="coerce")
        df_long[value_name] = np.where(df_long[value_name] > 1000, df_long[value_name] / 1e14, df_long[value_name])
    return df_long

def parse_price_series(s):
    s = s.astype(str)
    parts = s.str.split("E", n=1, expand=True)
    mant = parts[0].str.replace(",", ".", regex=False).str.replace(" ", "", regex=False)
    vals = pd.to_numeric(mant, errors="coerce") * 100.0
    return vals

def load_global_price_context(path):
    df = pd.read_csv(path, sep=";", dtype=str)
    df["months"] = pd.to_datetime(df["months"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["months"])
    df["Colombian Milds"] = parse_price_series(df["Colombian Milds"])
    df["Robustas"] = parse_price_series(df["Robustas"])
    df = df.set_index("months")
    agg = df.groupby(df.index.year).agg({"Colombian Milds": "mean", "Robustas": "mean"}).rename(columns={"Colombian Milds": "Preco_Milds_Global", "Robustas": "Preco_Robustas_Global"})
    agg.index.name = "Ano"
    return agg

def impute_by_column(df):
    df = df.apply(pd.to_numeric, errors="coerce")
    for col in df.columns:
        med = df[col].median(skipna=True)
        df[col] = df[col].fillna(med).ffill().bfill()
    return df

def main():
    path_prices = os.path.join(DATA_DIR, "prices-paid-to-growers.csv")
    inv_path = os.path.join(DATA_DIR, "inventories.csv")
    prod_path = os.path.join(DATA_DIR, "total-production.csv")
    cons_path = os.path.join(DATA_DIR, "domestic-consumption.csv")
    ind_prices_path = os.path.join(DATA_DIR, "indicator-prices.csv")

    y_long = impute_and_melt(path_prices, "Preco_Produtor")
    prod_long = impute_and_melt(prod_path, "Producao_Nacional")
    cons_long = impute_and_melt(cons_path, "Consumo_Nacional")
    inv_long = impute_and_melt(inv_path, "Estoque_Nacional")

    df_final = y_long.merge(prod_long, on=["Ano", "Pais"], how="left")
    df_final = df_final.merge(cons_long, on=["Ano", "Pais"], how="left")
    df_final = df_final.merge(inv_long, on=["Ano", "Pais"], how="left")

    df_final["Saldo_Nacional"] = df_final["Producao_Nacional"] - df_final["Consumo_Nacional"]
    df_final["Razao_Estoque_Consumo_Nacional"] = df_final["Estoque_Nacional"] / (df_final["Consumo_Nacional"] + 1e-9)

    df_global = load_global_price_context(ind_prices_path)
    df_final = df_final.merge(df_global, on="Ano", how="left")

    num_cols = [
        "Producao_Nacional",
        "Consumo_Nacional",
        "Estoque_Nacional",
        "Saldo_Nacional",
        "Preco_Milds_Global",
        "Preco_Robustas_Global",
    ]
    for col in num_cols:
        df_final[col] = pd.to_numeric(df_final[col], errors="coerce")
        df_final[col] = df_final[col].fillna(df_final[col].median())
        df_final[col] = np.log1p(df_final[col])

    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_final = df_final.dropna(subset=["Preco_Produtor"]) 

    dummies = pd.get_dummies(df_final["Pais"], prefix="Pais")
    X = pd.concat([
        df_final[[
            "Producao_Nacional",
            "Consumo_Nacional",
            "Saldo_Nacional",
            "Preco_Milds_Global",
            "Preco_Robustas_Global",
        ]].reset_index(drop=True),
        dummies.reset_index(drop=True)
    ], axis=1)
    y = df_final["Preco_Produtor"].reset_index(drop=True)

    if X.shape[0] == 0:
        print("Dataset vazio apos merges/limpeza. Verifique disponibilidade de anos/paises e arquivos.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "n_estimators": [50, 100, 200],
    }
    grid = GridSearchCV(
        estimator=XGBRegressor(objective="reg:squarederror", random_state=42),
        param_grid=param_grid,
        scoring="r2",
        cv=3,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    abs_err = np.abs(y_test - y_pred)
    denom = np.where(np.abs(y_test) > 0, np.abs(y_test), np.nan)
    tol_acc = np.nanmean((abs_err / denom <= 0.05).astype(float)) * 100

    print(f"Best Params: {grid.best_params_}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"Acuracia_5pct: {tol_acc:.2f}%")

    df_comp = pd.DataFrame({"Real": y_test, "Predito": y_pred, "Erro_Absoluto": abs_err}).sort_values("Erro_Absoluto", ascending=False).head(10)
    print(df_comp.to_string())

    fi = pd.DataFrame({"feature": X.columns, "importance": best_model.feature_importances_}).sort_values("importance", ascending=False)
    print(fi.to_string(index=False))

if __name__ == "__main__":
    main()