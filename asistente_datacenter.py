import cv2
import face_recognition as fr
import os
import numpy as np
from datetime import datetime

# Crear base de datos de personal autorizado
ruta = 'data/personal'
imagenes_personal = []
nombres_personal = []
lista_archivos = os.listdir(ruta)

for archivo in lista_archivos:
    imagen_actual = cv2.imread(f"{ruta}/{archivo}")
    if imagen_actual is None:
        print(f"No se pudo leer {archivo}, se omitirá.")
        continue
    imagenes_personal.append(imagen_actual)
    nombres_personal.append(os.path.splitext(archivo)[0])

print(f"Personal autorizado cargado: {nombres_personal}")

# --- Función para codificar imágenes ---
def codificar(imagenes):
    lista_codificada = []
    for imagen in imagenes:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        codificado = fr.face_encodings(imagen)[0]
        lista_codificada.append(codificado)
    return lista_codificada

codigos_codificados = codificar(imagenes_personal)

# --- Función para registrar accesos ---
def registrar_evento(tipo, persona, carpeta="data/registros"):
    os.makedirs(carpeta, exist_ok=True)
    archivo = os.path.join(carpeta, f"{tipo}.csv")

    if not os.path.exists(archivo):
        with open(archivo, "w") as f:
            f.write("Nombre,Hora,Fecha\n")

    with open(archivo, "a") as f:
        ahora = datetime.now()
        f.write(f"{persona},{ahora.strftime('%H:%M:%S')},{ahora.strftime('%Y-%m-%d')}\n")

# --- Función principal de reconocimiento ---
def control_acceso():
    print("=== Sistema de Control de Acceso al Data Center ===")
    camara = cv2.VideoCapture(0)

    if not camara.isOpened():
        print("No se pudo acceder a la cámara.")
        return

    while True:
        exito, frame = camara.read()
        if not exito:
            print("No se pudo capturar el video.")
            break

        # Reducir tamaño para mayor velocidad
        pequeño = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb = cv2.cvtColor(pequeño, cv2.COLOR_BGR2RGB)

        # Detección y codificación
        caras = fr.face_locations(rgb)
        codigos = fr.face_encodings(rgb, caras)

        for (top, right, bottom, left), cara_codif in zip(caras, codigos):
            distancias = fr.face_distance(codigos_codificados, cara_codif)
            if len(distancias) == 0:
                continue

            indice = np.argmin(distancias)
            nombre = "NO AUTORIZADO"
            color = (0, 0, 255)

            if distancias[indice] < 0.6:
                nombre = nombres_personal[indice]
                color = (0, 255, 0)
                registrar_evento("accesos", nombre)
            else:
                registrar_evento("intentos_no_aut", nombre)

            # Escalar coordenadas (ya que se redujo a 1/4)
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Dibujar recuadro y texto
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, nombre, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

        cv2.imshow("Control de Acceso - Presione 'q' para salir", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camara.release()
    cv2.destroyAllWindows()

# --- Inicio ---
if __name__ == "__main__":
    print("=== SISTEMA DE ACCESO AL DATA CENTER ===")
    control_acceso()
