import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# === CONFIGURACIÓN DE RUTAS ===
carpeta = "data/registros"
archivo_accesos = os.path.join(carpeta, "accesos.csv")
archivo_intentos = os.path.join(carpeta, "intentos_no_aut.csv")
salida_grafico = os.path.join(carpeta, "reporte_grafico.png")

# === VERIFICACIÓN DE ARCHIVOS ===
if not os.path.exists(archivo_accesos):
    print("No se encontraron registros de accesos.")
    exit()

# === LECTURA DE REGISTROS ===
accesos = pd.read_csv(archivo_accesos)
print("Registros de accesos cargados correctamente")

# Normalizar nombres de columnas (corrige errores de formato)
accesos.columns = [col.strip().capitalize() for col in accesos.columns]
if "Nombre" not in accesos.columns:
    accesos.columns = ["Nombre", "Hora", "Fecha"]

# Cargar intentos no autorizados (si existen)
if os.path.exists(archivo_intentos):
    intentos = pd.read_csv(archivo_intentos)
    intentos.columns = [col.strip().capitalize() for col in intentos.columns]
    if "Nombre" not in intentos.columns:
        intentos.columns = ["Nombre", "Hora", "Fecha"]
else:
    intentos = pd.DataFrame(columns=["Nombre", "Hora", "Fecha"])

# === ANÁLISIS DE DATOS ===
total_accesos = len(accesos)
total_intentos = len(intentos)
total_eventos = total_accesos + total_intentos
porcentaje_validos = (total_accesos / total_eventos * 100) if total_eventos > 0 else 0

# Conteo de accesos por persona
conteo_por_persona = accesos["Nombre"].value_counts()
persona_top = conteo_por_persona.idxmax() if not conteo_por_persona.empty else "N/A"
accesos_top = conteo_por_persona.max() if not conteo_por_persona.empty else 0

# === RESUMEN TEXTUAL ===
print("\n=== REPORTE DE CONTROL DE ACCESOS AL DATA CENTER ===")
print(f"Fecha del reporte: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("------------------------------------------------------")
print(f"Total de accesos válidos: {total_accesos}")
print(f"Total de intentos no autorizados: {total_intentos}")
print(f"Porcentaje de accesos válidos: {porcentaje_validos:.2f}%")
print(f"Persona con más ingresos: {persona_top} ({accesos_top} veces)")
print("------------------------------------------------------\n")

# === GENERACIÓN DE GRÁFICOS ===
plt.figure(figsize=(12, 6))

# Subplot 1: Barras (accesos por persona)
plt.subplot(1, 2, 1)
conteo_por_persona.plot(kind="bar", color="#1f77b4")
plt.title("Accesos válidos por persona", fontsize=13, fontweight="bold")
plt.xlabel("Personal autorizado")
plt.ylabel("Cantidad de ingresos")
plt.xticks(rotation=45, ha="right")

# Subplot 2: Torta (porcentaje válidos vs no autorizados)
plt.subplot(1, 2, 2)
plt.pie(
    [total_accesos, total_intentos],
    labels=["Válidos", "No autorizados"],
    autopct="%1.1f%%",
    startangle=90,
    colors=["#2ca02c", "#d62728"]
)
plt.title("Distribución de accesos", fontsize=13, fontweight="bold")

plt.suptitle(" Reporte de Accesos al Data Center", fontsize=15, fontweight="bold")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Guardar y mostrar
plt.savefig(salida_grafico)
plt.show()

# === GUARDAR RESUMEN CSV ===
reporte = {
    "Fecha": [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
    "Total_Accesos": [total_accesos],
    "Intentos_No_Aut": [total_intentos],
    "Porcentaje_Validos": [round(porcentaje_validos, 2)],
    "Top_Persona": [persona_top],
    "Top_Cantidad": [accesos_top]
}

df_reporte = pd.DataFrame(reporte)
reporte_path = os.path.join(carpeta, "reporte_resumen.csv")
df_reporte.to_csv(
    reporte_path,
    mode="a",
    index=False,
    header=not os.path.exists(reporte_path) or os.path.getsize(reporte_path) == 0
)


print(f" Reporte visual guardado en: {salida_grafico}")
print(" Resumen CSV actualizado en: data/registros/reporte_resumen.csv")
