import cv2
import face_recognition as fr

# Cargar imágenes
foto_control = fr.load_image_file('data/personal/giuliana_bonora.png')
foto_prueba = fr.load_image_file('data/personal/juan_perez.png')

# Pasar las imágenes a RGB
foto_control = cv2.cvtColor(foto_control, cv2.COLOR_BGR2RGB)
foto_prueba = cv2.cvtColor(foto_prueba, cv2.COLOR_BGR2RGB)

# Localizar y codificar las caras
lugar_cara_A = fr.face_locations(foto_control)[0]
cara_codificada_A = fr.face_encodings(foto_control)[0]

lugar_cara_B = fr.face_locations(foto_prueba)[0]
cara_codificada_B = fr.face_encodings(foto_prueba)[0]

# Dibujar rectángulos
cv2.rectangle(foto_control, (lugar_cara_A[3], lugar_cara_A[0]), (lugar_cara_A[1], lugar_cara_A[2]), (0, 255, 0), 2)
cv2.rectangle(foto_prueba, (lugar_cara_B[3], lugar_cara_B[0]), (lugar_cara_B[1], lugar_cara_B[2]), (0, 255, 0), 2)

# Comparación de imágenes
resultado = fr.compare_faces([cara_codificada_A], cara_codificada_B)
distancia = fr.face_distance([cara_codificada_A], cara_codificada_B)

print(f"Coincidencia: {resultado}")
print(f"Distancia: {distancia}")

cv2.putText(foto_prueba, f"{resultado} {distancia.round(2)}", (50, 50),
            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

# Mostrar imágenes
cv2.imshow('Foto Control', foto_control)
cv2.imshow('Foto Prueba', foto_prueba)
cv2.waitKey(0)



