import cv2
import mediapipe as mp
from deepface import DeepFace

# Inicializar Mediapipe y DeepFace
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Capturar video desde la cámara
cap = cv2.VideoCapture(0)  # Cambia el índice si es necesario

# Mapeo de emociones de inglés a español
emotion_translation = {
    'angry': 'enfadado',
    'disgust': 'desagrado',
    'fear': 'miedo',
    'happy': 'feliz',
    'sad': 'triste',
    'surprise': 'sorpresa',
    'neutral': 'neutral'
}

# Inicializar el modelo de Mediapipe
with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        resized_frame = cv2.resize(rgb_frame, (224, 224))

        try:
            emotion_result = DeepFace.analyze(resized_frame, actions=['emotion'], enforce_detection=True)
            print(emotion_result)  # Imprimir resultados para depuración
            dominant_emotion = emotion_result[0]['dominant_emotion']
            # Traducir la emoción a español
            dominant_emotion_spanish = emotion_translation.get(dominant_emotion, "No detectado")
        except Exception as e:
            print(f"Error en la detección de emociones: {e}")
            dominant_emotion_spanish = "No detectado"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                           mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                                           mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1))

        # Mostrar la emoción dominante en español
        cv2.putText(frame, f'Emocion: {dominant_emotion_spanish}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Mostrar el frame resultante
        cv2.imshow('Face Mesh & Emotion Detection', frame)

        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()

