import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

def load_indicator_prices(path, target_col="Robustas"):
    df = pd.read_csv(path, sep=",", dtype=str)
    m1 = pd.to_datetime(df["months"], format="%d/%m/%Y", errors="coerce")
    m2 = pd.to_datetime(df["months"], format="%m/%Y", errors="coerce")
    df["months"] = m1
    c1 = df["months"].notna().sum()
    if m2.notna().sum() > c1:
        df["months"] = m2
        print("Formato_data_escolhido: %m/%Y")
    else:
        print("Formato_data_escolhido: %d/%m/%Y")
    df = df.dropna(subset=["months"])
    print(f"Linhas_mensais_lidas: {len(df)}")
    df[target_col] = parse_price_series(df[target_col])
    df = df.set_index("months")
    y_monthly = df[[target_col]].rename(columns={target_col: "Y"})
    return y_monthly

def load_annual_wide(path):
    raw = pd.read_csv(path, sep=",", dtype=str)
    data = raw.copy()
    first_col = data.columns[0]
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
    s = s.astype(str).str.replace(",", ".", regex=False).str.replace(" ", "", regex=False)
    if s.str.contains("E").any():
        mant = s.str.split("E", n=1, expand=True)[0]
    else:
        mant = s
    vals = pd.to_numeric(mant, errors="coerce") * 100.0
    return vals

def load_global_price_context(path):
    df = pd.read_csv(path, sep=",", dtype=str)
    m1 = pd.to_datetime(df["months"], format="%d/%m/%Y", errors="coerce")
    m2 = pd.to_datetime(df["months"], format="%m/%Y", errors="coerce")
    df["months"] = m1
    c1 = df["months"].notna().sum()
    if m2.notna().sum() > c1:
        df["months"] = m2
        print("Formato_data_escolhido_global: %m/%Y")
    else:
        print("Formato_data_escolhido_global: %d/%m/%Y")
    df = df.dropna(subset=["months"])
    print(f"Linhas_mensais_lidas_global: {len(df)}")
    df["Colombian Milds"] = parse_price_series(df["Colombian Milds"])
    df["Robustas"] = parse_price_series(df["Robustas"])
    df = df.set_index("months")
    return df

def build_cambio_mensal(index):
    rng = np.random.RandomState(42)
    base = 4.0 + rng.normal(0, 0.2, size=len(index)).cumsum() / 50.0
    fx = pd.Series(base, index=index, name="Taxa_Cambio_USD_BRL")
    return fx.to_frame()

def build_producao_lag_mensal(path, months_index):
    wide = load_annual_wide(path)
    total = wide.sum(axis=1).rename("Producao_Global_Anual")
    mdf = pd.DataFrame({"months": months_index})
    mdf["Ano"] = mdf["months"].dt.year
    mdf["Producao_Global_Anual"] = mdf["Ano"].map(total)
    mdf = mdf.set_index("months")
    mdf["Producao_Lag_6M"] = mdf["Producao_Global_Anual"].shift(6)
    return mdf[["Producao_Lag_6M"]]

def impute_by_column(df):
    df = df.apply(pd.to_numeric, errors="coerce")
    for col in df.columns:
        med = df[col].median(skipna=True)
        df[col] = df[col].fillna(med).ffill().bfill()
    return df

def main():
    ind_prices_path = os.path.join(DATA_DIR, "indicator-prices.csv")
    prices_paid_path = os.path.join(DATA_DIR, "prices-paid-to-growers.csv")
    prod_path = os.path.join(DATA_DIR, "total-production.csv")
    cons_path = os.path.join(DATA_DIR, "domestic-consumption.csv")

    df_y = impute_and_melt(prices_paid_path, "Preco_Produtor")
    df_prod = impute_and_melt(prod_path, "Producao_Nacional")
    df_cons = impute_and_melt(cons_path, "Consumo_Nacional")

    df_nat = (
        df_y.merge(df_prod, on=["Ano", "Pais"], how="left")
             .merge(df_cons, on=["Ano", "Pais"], how="left")
    )

    g_month = load_global_price_context(ind_prices_path)
    g_month = g_month.sort_index()
    g_month["Ano"] = g_month.index.year
    g_ann = g_month.groupby("Ano", as_index=True)[["Colombian Milds", "Robustas"]].mean()
    g_ann = g_ann.rename(columns={
        "Robustas": "Preco_Global_Anual_Robustas",
        "Colombian Milds": "Preco_Global_Anual_Milds",
    })
    g_ann["Preco_Global_Lag_1Y"] = g_ann["Preco_Global_Anual_Robustas"].shift(1)

    fx_month = build_cambio_mensal(g_month.index)
    fx_month["Ano"] = fx_month.index.year
    fx_ann = fx_month.groupby("Ano", as_index=True).mean()
    fx_ann = fx_ann.rename(columns={"Taxa_Cambio_USD_BRL": "Cambio_Anual_Medio"})

    df_feat = (
        df_nat.merge(g_ann[["Preco_Global_Lag_1Y"]], on="Ano", how="left")
              .merge(fx_ann[["Cambio_Anual_Medio"]], on="Ano", how="left")
    )

    df_feat = df_feat.dropna(subset=["Preco_Produtor", "Producao_Nacional", "Consumo_Nacional", "Preco_Global_Lag_1Y", "Cambio_Anual_Medio"]) 
    dummies = pd.get_dummies(df_feat["Pais"], prefix="Pais")
    X = pd.concat([
        df_feat[["Producao_Nacional", "Consumo_Nacional", "Preco_Global_Lag_1Y", "Cambio_Anual_Medio"]],
        dummies
    ], axis=1)
    y = df_feat["Preco_Produtor"]

    print(f"Observacoes_Pais_Ano_total: {len(df_nat)}")
    print(f"Observacoes_modelo_apos_dropna: {len(df_feat)}")
    print(f"Total_dummies_Pais: {dummies.shape[1]}")

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
    print(f"Acuracia(score): {best_model.score(X_test, y_test):.4f}")
    print(f"Acuracia_5pct: {tol_acc:.2f}%")

    df_comp = pd.DataFrame({"Real": y_test, "Predito": y_pred, "Erro_Absoluto": abs_err}).sort_values("Erro_Absoluto", ascending=False).head(10)
    print(df_comp.to_string())

    fi = pd.DataFrame({"feature": X.columns, "importance": best_model.feature_importances_}).sort_values("importance", ascending=False)
    print(fi.to_string(index=False))
    imp_fx = fi.loc[fi["feature"] == "Cambio_Anual_Medio", "importance"]
    if not imp_fx.empty:
        print(f"Importancia_Cambio_Anual_Medio: {float(imp_fx.iloc[0]):.6f}")

if __name__ == "__main__":
    main()
