import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

RUTA_REGISTROS = "data/registros"
ARCHIVO_AUTORIZADOS = os.path.join(RUTA_REGISTROS, "accesos_autorizados.csv")

def _parse_hf(x: str):
    try:
        return datetime.strptime(x, "%H:%M:%S del %d-%m-%Y")
    except Exception:
        return None

def generar_reportes():
    if not os.path.exists(ARCHIVO_AUTORIZADOS):
        print(f"No existe {ARCHIVO_AUTORIZADOS}")
        return

    df = pd.read_csv(ARCHIVO_AUTORIZADOS, dtype=str, keep_default_na=False)
    if df.empty:
        print("No hay datos para graficar.")
        return

    # Normalizar columnas esperadas
    for col in ["Nombre", "Accion", "Hora_Fecha", "Duracion_min"]:
        if col not in df.columns:
            df[col] = ""

    # Timestamp usable
    df["dt"] = df["Hora_Fecha"].apply(_parse_hf)
    df = df.dropna(subset=["dt"]).copy()

    # 1) Ingresos por persona (barras)
    df_ing = df[df["Accion"] == "ingreso"].copy()
    ingresos_por_persona = df_ing["Nombre"].value_counts().sort_values(ascending=False)
    if not ingresos_por_persona.empty:
        plt.figure(figsize=(9,5))
        ingresos_por_persona.plot(kind="bar")
        plt.title("Ingresos por persona")
        plt.xlabel("Persona"); plt.ylabel("Ingresos")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout(); plt.show()
    else:
        print("No hay ingresos para graficar.")

    # 2) Promedio de permanencia (barras, min)
    df_egr = df[df["Accion"] == "egreso"].copy()
    df_egr["Duracion_min"] = pd.to_numeric(df_egr["Duracion_min"], errors="coerce")
    df_egr = df_egr.dropna(subset=["Duracion_min"])
    if not df_egr.empty:
        prom_permanencia = df_egr.groupby("Nombre")["Duracion_min"].mean().sort_values(ascending=False)
        plt.figure(figsize=(9,5))
        prom_permanencia.plot(kind="bar")
        plt.title("Promedio de permanencia (min)")
        plt.xlabel("Persona"); plt.ylabel("Minutos (promedio)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout(); plt.show()
    else:
        print("No hay egresos con duración para graficar.")

    # 3) Evolución diaria (línea)
    if not df_ing.empty:
        df_ing["Fecha"] = df_ing["dt"].dt.date
        ingresos_por_dia = df_ing["Fecha"].value_counts().sort_index()
        if not ingresos_por_dia.empty:
            plt.figure(figsize=(9,5))
            plt.plot(ingresos_por_dia.index.astype(str), ingresos_por_dia.values, marker="o")
            plt.title("Evolución diaria de ingresos")
            plt.xlabel("Fecha"); plt.ylabel("Cantidad de ingresos")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout(); plt.show()
        else:
            print("No hay ingresos con fecha válida para graficar.")

if __name__ == "__main__":
    generar_reportes()



