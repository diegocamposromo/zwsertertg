import cv2
import mediapipe as mp

# Inicializar el módulo de MediaPipe y los detectores de la mano y el dibujo de la mano
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def main():
    # Inicializar la cámara
    cap = cv2.VideoCapture(0)

    # Inicializar el detector de manos
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
        
        while True:
            # Leer cada fotograma de la cámara
            ret, frame = cap.read()
            
            # Convertir la imagen de BGR a RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Realizar la detección de manos
            results = hands.process(image)

            # Verificar si se detectó alguna mano
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Obtener la posición del pulgar
                    thumb_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    
                    # Obtener la altura de la imagen (altura de la pantalla)
                    image_height, image_width, _ = frame.shape
                    
                    # Determinar si el pulgar está arriba o abajo
                    if thumb_landmark.y * image_height < image_height / 2:
                        thumb_position = "Pulgar hacia arriba"
                    else:
                        thumb_position = "Pulgar hacia abajo"

                    # Dibujar el resultado en la imagen
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    )
                    
                    # Mostrar el texto con la posición del pulgar en la esquina superior izquierda
                    cv2.putText(frame, thumb_position, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Mostrar la imagen procesada en una ventana
            cv2.imshow('Thumb Detection', frame)
            
            # Romper el bucle si se presiona la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Liberar los recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()