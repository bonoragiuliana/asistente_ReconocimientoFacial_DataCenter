import cv2
import face_recognition as fr
import os
import numpy as np
import pandas as pd
from datetime import datetime

# importar la fuente única de gráficos
from reporte_datacenter import generar_reportes

RUTA_PERSONAL = "data/personal"  # fotos del personal. El nombre del archivo (sin extensión) es el "Nombre" que se registra
RUTA_REGISTROS = "data/registros"
ARCHIVO_PERSONAL = "data/personal.csv"  # permisos, con nombre, rol, permiso (SI/NO)
ARCHIVO_AUTORIZADOS = os.path.join(RUTA_REGISTROS, "accesos_autorizados.csv")  # registra ingresos/egresos válidos
ARCHIVO_NO_AUTORIZADOS = os.path.join(RUTA_REGISTROS, "accesos_no_autorizados.csv")  # intentos inválidos

# si las carpetas no existen, crearlas (evita errores al iniciar en limpio)
os.makedirs(RUTA_PERSONAL, exist_ok=True)
os.makedirs(RUTA_REGISTROS, exist_ok=True)

# ejemplo por defecto si no existe el catálogo de personal
if not os.path.exists(ARCHIVO_PERSONAL):
    with open(ARCHIVO_PERSONAL, "w") as f:
        f.write("Nombre,Rol,Permiso\nGiulianaBonora,Administradora,SI\n")

# encabezados de registros solo si los archivos no existen
if not os.path.exists(ARCHIVO_AUTORIZADOS):
    with open(ARCHIVO_AUTORIZADOS, "w") as f:
        f.write("Nombre,Accion,Hora_Fecha,Duracion_min\n")

if not os.path.exists(ARCHIVO_NO_AUTORIZADOS):
    with open(ARCHIVO_NO_AUTORIZADOS, "w") as f:
        f.write("Nombre,Tipo,Hora_Fecha\n")

# lee el CSV de permisos y lo pasa a diccionario {Nombre: "SI"/"NO"}
permisos_df = pd.read_csv(ARCHIVO_PERSONAL)
permisos_dict = {row["Nombre"]: row["Permiso"].strip().upper() for _, row in permisos_df.iterrows()}

# carga de imágenes del personal. El nombre del archivo coincide con "Nombre" del CSV.
imagenes_personal = []
nombres_personal = []
for archivo in os.listdir(RUTA_PERSONAL):
    if archivo.lower().endswith((".jpg", ".png", ".jpeg")):
        imagen_actual = cv2.imread(os.path.join(RUTA_PERSONAL, archivo))
        if imagen_actual is None:
            print(f"No se pudo leer {archivo}, se omitirá.")
            continue
        imagenes_personal.append(imagen_actual)
        nombres_personal.append(os.path.splitext(archivo)[0])

print(f"Personal cargado: {nombres_personal}")

# Codificar imagenes
def codificar(imagenes):
    lista_codificada = []
    for imagen in imagenes:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        codificado = fr.face_encodings(imagen)[0]
        lista_codificada.append(codificado)
    return lista_codificada

codigos_codificados = codificar(imagenes_personal)

# formateo de hora y fecha
def ahora_str():
    return datetime.now().strftime("%H:%M:%S del %d-%m-%Y")

def parse_hora_fecha(hf: str):
    try:
        return datetime.strptime(hf, "%H:%M:%S del %d-%m-%Y")
    except Exception:
        return None

# accesos autorizados (ingreso/egreso presionando tecla I/E)
def registrar_acceso_autorizado(persona: str, es_ingreso: bool) -> bool:
    ahora = datetime.now()
    hf_actual = ahora.strftime("%H:%M:%S del %d-%m-%Y")

    if es_ingreso:
        with open(ARCHIVO_AUTORIZADOS, "a") as f:
            f.write(f"{persona},ingreso,{hf_actual},\n")
        print(f"Se habilita ingreso para {persona} a las {hf_actual}")
        return True

    # Egreso: validar que haya un ingreso abierto
    try:
        df = pd.read_csv(ARCHIVO_AUTORIZADOS, dtype=str, keep_default_na=False)
    except Exception as e:
        print(f"No se pudo leer {ARCHIVO_AUTORIZADOS}: {e}")
        return False

    # normalizar columnas
    for col in ["Nombre", "Accion", "Hora_Fecha", "Duracion_min"]:
        if col not in df.columns:
            df[col] = ""

    df_p = df[df["Nombre"] == persona].copy()
    last_ing = df_p[df_p["Accion"] == "ingreso"].tail(1)
    last_egr = df_p[df_p["Accion"] == "egreso"].tail(1)

    if last_ing.empty:
        print(f"No se encontró un ingreso previo para {persona}.")
        return False

    ing_dt = parse_hora_fecha(last_ing["Hora_Fecha"].iloc[0])
    egr_dt = parse_hora_fecha(last_egr["Hora_Fecha"].iloc[0]) if not last_egr.empty else None

    ingreso_abierto = (ing_dt is not None) and (egr_dt is None or ing_dt > egr_dt)
    if not ingreso_abierto:
        print(f"{persona} no tiene un ingreso previo como para registrar un egreso.")
        return False

    # duración en minutos con 2 decimales
    duracion_min = round(max(0.0, (ahora - ing_dt).total_seconds() / 60.0), 2)

    with open(ARCHIVO_AUTORIZADOS, "a") as f:
        f.write(f"{persona},egreso,{hf_actual},{duracion_min:.2f}\n")

    print(f"Se registra egreso de {persona} a las {hf_actual} (Tiempo dentro del DataCenter: {duracion_min:.2f} min)")
    return True

# Accesos NO autorizados (solo cuando se presiona I)
def registrar_no_autorizado(nombre: str, tipo: str):
    fecha = ahora_str()
    with open(ARCHIVO_NO_AUTORIZADOS, "a") as f:
        f.write(f"{nombre},{tipo},{fecha}\n")
    print(f"Se niega acceso a {nombre} ({tipo}) a las {fecha}")

# Reconocimiento en tiempo real
def control_acceso_continuo():
    print(" SISTEMA DE CONTROL DE ACCESO AL DATA CENTER ")
    print("Presiona I para registrar ingreso, E para registrar egreso, Q para salir.\n")

    camara = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not camara.isOpened():
        print("No se pudo acceder a la cámara. Revisa permisos en Seguridad y Privacidad.")
        return

    # el modo indica la acción pendiente a ejecutar cuando se detecte un rostro.
    modo = None  # None | "ingreso" | "egreso"

    while True:
        exito, frame = camara.read()
        if not exito:
            print("No se pudo capturar el video.")
            break

        # face_recognition trabaja en RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        caras = fr.face_locations(rgb)
        codigos = fr.face_encodings(rgb, caras)

        for (top, right, bottom, left), cara_codif in zip(caras, codigos):
            distancias = fr.face_distance(codigos_codificados, cara_codif)
            if len(distancias) == 0:
                continue

            indice = np.argmin(distancias)
            match = distancias[indice] < 0.6  # umbral de similitud

            nombre = "Sin registro"
            color = (0, 0, 255)  # rojo por defecto

            if match:
                nombre = nombres_personal[indice]
                permiso = permisos_dict.get(nombre, "NO")

                if permiso == "SI":
                    color = (0, 255, 0)  # verde
                    if modo == "ingreso":
                        registrar_acceso_autorizado(nombre, True)
                        modo = None
                    elif modo == "egreso":
                        _ = registrar_acceso_autorizado(nombre, False)  # valida ingreso abierto
                        modo = None
                else:
                    color = (0, 0, 255)  # rojo: conocido sin permiso
                    if modo == "ingreso":
                        registrar_no_autorizado(nombre, "Sin permiso")
                        modo = None
            else:
                # rostro desconocido
                if modo == "ingreso":
                    registrar_no_autorizado("Sin registro", "No registrado")
                    modo = None

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, nombre, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

        # instrucciones en cámara
        cv2.putText(frame, "Presiona I=Ingreso | E=Egreso | Q=Salir", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Control de Acceso - Data Center", frame)

        # Teclado no bloqueante
        tecla = cv2.waitKey(10) & 0xFF
        if tecla == ord('q'):
            print("Saliendo del sistema y generando reportes...")
            break
        elif tecla == ord('i'):
            print("Modo ingreso activado.")
            modo = "ingreso"
        elif tecla == ord('e'):
            print("Modo egreso activado.")
            modo = "egreso"

    camara.release()
    cv2.destroyAllWindows()
    # llamar al módulo único de reportes
    generar_reportes()

# punto de entrada
if __name__ == "__main__":
    control_acceso_continuo()




