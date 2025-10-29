import cv2
import face_recognition as fr
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from reporte_datacenter import generar_graficos_tres


# === CONFIGURACIÓN GENERAL ===
RUTA_PERSONAL = "data/personal"
RUTA_REGISTROS = "data/registros"
ARCHIVO_PERSONAL = "data/personal.csv"
ARCHIVO_AUTORIZADOS = os.path.join(RUTA_REGISTROS, "accesos_autorizados.csv")
ARCHIVO_NO_AUTORIZADOS = os.path.join(RUTA_REGISTROS, "accesos_no_autorizados.csv")

# === CREAR ESTRUCTURA SI NO EXISTE ===
os.makedirs(RUTA_PERSONAL, exist_ok=True)
os.makedirs(RUTA_REGISTROS, exist_ok=True)

# Crear archivo de personal si no existe
if not os.path.exists(ARCHIVO_PERSONAL):
    with open(ARCHIVO_PERSONAL, "w") as f:
        f.write("Nombre,Rol,Permiso\nGiulianaBonora,Administradora,SI\n")

# Crear archivos de registro si no existen
if not os.path.exists(ARCHIVO_AUTORIZADOS):
    with open(ARCHIVO_AUTORIZADOS, "w") as f:
        f.write("Nombre,Accion,Hora_Fecha,Duracion_min\n")

if not os.path.exists(ARCHIVO_NO_AUTORIZADOS):
    with open(ARCHIVO_NO_AUTORIZADOS, "w") as f:
        f.write("Nombre,Tipo,Hora_Fecha\n")

# === CARGAR PERSONAL Y PERMISOS ===
permisos_df = pd.read_csv(ARCHIVO_PERSONAL)
permisos_dict = {row["Nombre"]: row["Permiso"].strip().upper() for _, row in permisos_df.iterrows()}

# === CARGAR IMÁGENES DEL PERSONAL ===
imagenes_personal = []
nombres_personal = []
for archivo in os.listdir(RUTA_PERSONAL):
    if archivo.lower().endswith((".jpg", ".png", ".jpeg")):
        imagen_actual = cv2.imread(os.path.join(RUTA_PERSONAL, archivo))
        if imagen_actual is None:
            print(f"⚠️ No se pudo leer {archivo}, se omitirá.")
            continue
        imagenes_personal.append(imagen_actual)
        nombres_personal.append(os.path.splitext(archivo)[0])

print(f"👥 Personal cargado: {nombres_personal}")

# === FUNCIÓN PARA CODIFICAR IMÁGENES ===
def codificar(imagenes):
    lista_codificada = []
    for imagen in imagenes:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        codificado = fr.face_encodings(imagen)[0]
        lista_codificada.append(codificado)
    return lista_codificada

codigos_codificados = codificar(imagenes_personal)

# === UTILIDADES ===
def ahora_str():
    return datetime.now().strftime("%H:%M:%S del %d-%m-%Y")

def parse_hora_fecha(hf: str) -> datetime | None:
    try:
        return datetime.strptime(hf, "%H:%M:%S del %d-%m-%Y")
    except Exception:
        return None

# === REGISTRAR ACCESO AUTORIZADO (INGRESO/EGRESO) ===
def registrar_acceso_autorizado(persona: str, es_ingreso: bool) -> bool:
    ahora = datetime.now()
    hf_actual = ahora.strftime("%H:%M:%S del %d-%m-%Y")

    if es_ingreso:
        # Agregamos un registro de ingreso
        with open(ARCHIVO_AUTORIZADOS, "a") as f:
            f.write(f"{persona},ingreso,{hf_actual},\n")
        print(f"✅ Se habilita ingreso para {persona} a las {hf_actual}")
        return True

    # Egreso: solo si hay un ingreso abierto (último ingreso posterior al último egreso)
    try:
        df = pd.read_csv(ARCHIVO_AUTORIZADOS, dtype=str, keep_default_na=False)
    except Exception as e:
        print(f"⚠️ No se pudo leer {ARCHIVO_AUTORIZADOS}: {e}")
        return False

    # Normalizar columnas
    for col in ["Nombre", "Accion", "Hora_Fecha", "Duracion_min"]:
        if col not in df.columns:
            df[col] = ""

    df_p = df[df["Nombre"] == persona].copy()
    last_ing = df_p[df_p["Accion"] == "ingreso"].tail(1)
    last_egr = df_p[df_p["Accion"] == "egreso"].tail(1)

    if last_ing.empty:
        print(f"⚠️ No se encontró un ingreso previo para {persona}.")
        return False

    ing_dt = parse_hora_fecha(last_ing["Hora_Fecha"].iloc[0])
    egr_dt = parse_hora_fecha(last_egr["Hora_Fecha"].iloc[0]) if not last_egr.empty else None

    # Condición de “ingreso abierto”: último ingreso existe y es posterior al último egreso
    ingreso_abierto = (ing_dt is not None) and (egr_dt is None or ing_dt > egr_dt)
    if not ingreso_abierto:
        print(f"⚠️ {persona} no tiene un ingreso abierto. No se registra egreso.")
        return False

    # Calcular duración en MINUTOS (dos decimales)
    duracion_min = round(max(0.0, (ahora - ing_dt).total_seconds() / 60.0), 2)

    # Registramos un nuevo renglón de EGRESO (no reescribimos filas anteriores)
    with open(ARCHIVO_AUTORIZADOS, "a") as f:
        f.write(f"{persona},egreso,{hf_actual},{duracion_min:.2f}\n")

    print(f"📤 Se registra egreso de {persona} a las {hf_actual} (Duración: {duracion_min:.2f} min)")
    return True

# === REGISTRAR ACCESO NO AUTORIZADO (solo al presionar I) ===
def registrar_no_autorizado(nombre: str, tipo: str):
    fecha = ahora_str()
    with open(ARCHIVO_NO_AUTORIZADOS, "a") as f:
        f.write(f"{nombre},{tipo},{fecha}\n")
    print(f"🚫 Se niega acceso a {nombre} ({tipo}) a las {fecha}")

# === GRÁFICO: TRÁFICO HORARIO ===
def generar_graficos_tres():
    try:
        df = pd.read_csv(ARCHIVO_AUTORIZADOS, dtype=str, keep_default_na=False)
        if df.empty:
            print("⚠️ No hay datos para generar gráficos.")
            return

        # Normalizar columnas claves
        for col in ["Nombre", "Accion", "Hora_Fecha", "Duracion_min"]:
            if col not in df.columns:
                df[col] = ""

        # --- Parseo de fecha-hora ---
        def parse_hf(x):
            try:
                return datetime.strptime(x, "%H:%M:%S del %d-%m-%Y")
            except Exception:
                return None

        df["dt"] = df["Hora_Fecha"].apply(parse_hf)
        df = df.dropna(subset=["dt"]).copy()

        # ===============================
        # 1) INGRESOS POR PERSONA
        # ===============================
        df_ing = df[df["Accion"] == "ingreso"].copy()
        ingresos_por_persona = df_ing["Nombre"].value_counts().sort_values(ascending=False)

        if not ingresos_por_persona.empty:
            plt.figure(figsize=(9,5))
            ingresos_por_persona.plot(kind="bar")
            plt.title("Ingresos por persona")
            plt.xlabel("Persona"); plt.ylabel("Ingresos")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()
        else:
            print("ℹ️ No hay ingresos para graficar (Ingresos por persona).")

        # ===============================
        # 2) PROMEDIO DE PERMANENCIA (min)
        # ===============================
        df_egr = df[df["Accion"] == "egreso"].copy()
        # Forzar numérico
        df_egr["Duracion_min"] = pd.to_numeric(df_egr["Duracion_min"], errors="coerce")
        df_egr = df_egr.dropna(subset=["Duracion_min"])

        if not df_egr.empty:
            prom_permanencia = df_egr.groupby("Nombre")["Duracion_min"].mean().sort_values(ascending=False)
            plt.figure(figsize=(9,5))
            prom_permanencia.plot(kind="bar")
            plt.title("Promedio de permanencia (min)")
            plt.xlabel("Persona"); plt.ylabel("Minutos (promedio)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()
        else:
            print("ℹ️ No hay egresos con duración para graficar (Promedio de permanencia).")

        # ===============================
        # 3) EVOLUCIÓN DIARIA (ingresos por día)
        # ===============================
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
                plt.tight_layout()
                plt.show()
            else:
                print("ℹ️ No hay ingresos con fecha válida para graficar (Evolución diaria).")
        else:
            print("ℹ️ No hay ingresos para graficar (Evolución diaria).")

    except Exception as e:
        print(f"⚠️ No se pudieron generar los gráficos: {e}")


# === RECONOCIMIENTO FACIAL CONTINUO ===
def control_acceso_continuo():
    print("=== SISTEMA DE CONTROL DE ACCESO AL DATA CENTER ===")
    print("Presiona I para registrar ingreso, E para registrar egreso, Q para salir.\n")

    camara = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not camara.isOpened():
        print("❌ No se pudo acceder a la cámara. Revisa permisos en Seguridad y Privacidad.")
        return

    # modo = None | "ingreso" | "egreso"
    modo = None

    while True:
        exito, frame = camara.read()
        if not exito:
            print("No se pudo capturar el video.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        caras = fr.face_locations(rgb)
        codigos = fr.face_encodings(rgb, caras)

        for (top, right, bottom, left), cara_codif in zip(caras, codigos):
            distancias = fr.face_distance(codigos_codificados, cara_codif)
            if len(distancias) == 0:
                continue

            indice = np.argmin(distancias)
            match = distancias[indice] < 0.6

            nombre = "Sin registro"
            color = (0, 0, 255)  # rojo por defecto

            if match:
                nombre = nombres_personal[indice]
                permiso = permisos_dict.get(nombre, "NO")

                if permiso == "SI":
                    color = (0, 255, 0)  # verde autorizado
                    if modo == "ingreso":
                        registrar_acceso_autorizado(nombre, True)
                        modo = None
                    elif modo == "egreso":
                        # Egreso solo si hubo ingreso abierto
                        registrado = registrar_acceso_autorizado(nombre, False)
                        modo = None
                        # Si no registrado, ya imprime la advertencia adentro
                else:
                    color = (0, 0, 255)
                    if modo == "ingreso":
                        registrar_no_autorizado(nombre, "Sin permiso")
                        modo = None
            else:
                if modo == "ingreso":
                    registrar_no_autorizado("Sin registro", "No registrado")
                    modo = None

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, nombre, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

        cv2.putText(frame, "Presiona I=Ingreso | E=Egreso | Q=Salir", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Control de Acceso - Data Center", frame)

        tecla = cv2.waitKey(10) & 0xFF
        if tecla == ord('q'):
            print("Saliendo del sistema y generando gráfico...")
            break
        elif tecla == ord('i'):
            print("➡️  Modo ingreso activado.")
            modo = "ingreso"
        elif tecla == ord('e'):
            print("⬅️  Modo egreso activado.")
            modo = "egreso"

    camara.release()
    cv2.destroyAllWindows()
    generar_graficos_tres()

# === INICIO ===
if __name__ == "__main__":
    control_acceso_continuo()



