import cv2
import mediapipe as mp
import math

# Mediapipe의 얼굴 검출 모듈 초기화
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 웹캠 초기화
cap = cv2.VideoCapture(0)

def calculate_yaw_angle(detection):
    landmarks = detection.location_data.relative_keypoints

    # 왼쪽 눈, 오른쪽 눈, 코의 landmark 인덱스
    landmarks = detection.location_data.relative_keypoints

    # 왼쪽 눈, 오른쪽 눈, 코의 landmark 인덱스
    left_eye = landmarks[mp_face_detection.FaceKeyPoint.LEFT_EYE]
    right_eye = landmarks[mp_face_detection.FaceKeyPoint.RIGHT_EYE]
    nose = landmarks[mp_face_detection.FaceKeyPoint.NOSE_TIP]

    # 두 눈 중심점 계산
    eye_center_x = (left_eye.x + right_eye.x) / 2
    eye_center_y = (left_eye.y + right_eye.y) / 2

    # 코 좌표
    nose_x = nose.x
    nose_y = nose.y

    # 코를 기준으로 두 눈의 중심점 사이의 각도 계산
    angle_rad = math.atan2(eye_center_y - nose_y, eye_center_x - nose_x)
    angle_deg = math.degrees(angle_rad) + 90  # +90을 추가하여 기준을 맞춤

    return angle_deg

# Mediapipe 얼굴 검출 모델 초기화
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR 이미지를 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 얼굴 검출 수행
        results = face_detection.process(image)

        # 검출된 얼굴이 있는 경우
        if results.detections:
            for detection in results.detections:
                # 얼굴 랜드마크 추출
                angle_deg = calculate_yaw_angle(detection)
                # 화면에 얼굴 방향 각도 출력
                cv2.putText(image, f"Yaw Angle: {angle_deg:.2f} degrees", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # 검출된 얼굴 주위에 사각형 그리기
                mp_drawing.draw_detection(image, detection)

        # BGR 이미지로 변환하여 화면에 출력
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('MediaPipe Face Detection', image)

        # 'q' 키를 누를 때까지 대기
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 웹캠과 모든 창 닫기
cap.release()
cv2.destroyAllWindows()
