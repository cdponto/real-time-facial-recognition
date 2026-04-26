import cv2
import os

# Use laptop webcam
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error: Could not open laptop camera.")
    exit()

facedetect = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

count = 0

# Fixed folder name
path = "images/teach-machine/suzanne"

# Create folder if it doesn't exist
os.makedirs(path, exist_ok=True)

while True:
    ret, frame = video.read()

    if not ret or frame is None:
        print("Error: Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1

        # Naming: image1.jpg → image500.jpg
        filename = f"{path}/image{count}.jpg"
        print("Creating:", filename)

        face_img = frame[y:y+h, x:x+w]
        cv2.imwrite(filename, face_img)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Laptop Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q") or count >= 500:
        break

video.release()
cv2.destroyAllWindows()