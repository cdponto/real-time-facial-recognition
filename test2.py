import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model


np.set_printoptions(suppress=True)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATHS = [
    BASE_DIR / "keras_model.h5",
    BASE_DIR / "keras_Model.h5",
]
LABELS_PATH = BASE_DIR / "labels.txt"
MIN_CONFIDENCE = 0.60
MIN_MARGIN = 0.10
FACE_MARGIN = 0.20


def resolve_model_path():
    for model_path in MODEL_PATHS:
        if model_path.exists():
            return model_path
    raise FileNotFoundError("Could not find keras_model.h5 or keras_Model.h5")


MODEL_PATH = resolve_model_path()

with LABELS_PATH.open("r", encoding="utf-8") as labels_file:
    class_names = [line.strip() for line in labels_file if line.strip()]

display_names = [name.split(" ", 1)[1] if " " in name else name for name in class_names]


def build_compatible_model():
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        alpha=0.35,
        include_top=False,
        weights=None,
        pooling="avg",
    )

    # Match older Teachable Machine export layer names.
    base_model._name = "sequential_1"

    rebuilt_model = Sequential(
        [
            base_model,
            Dense(100, activation="relu", name="dense_Dense1"),
            Dense(len(class_names), activation="softmax", use_bias=False, name="dense_Dense2"),
        ]
    )
    rebuilt_model.build((None, 224, 224, 3))
    rebuilt_model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=False)
    return rebuilt_model


def load_tm_model():
    try:
        return load_model(MODEL_PATH, compile=False)
    except Exception:
        return build_compatible_model()


model = load_tm_model()
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def preprocess_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb_frame.shape[:2]
    scale = max(224 / width, 224 / height)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    resized_frame = cv2.resize(
        rgb_frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )

    start_x = max(0, (resized_width - 224) // 2)
    start_y = max(0, (resized_height - 224) // 2)
    image_array = resized_frame[start_y:start_y + 224, start_x:start_x + 224]

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    return data


def extract_face_region(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return None, None

    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
    margin_x = int(w * FACE_MARGIN)
    margin_y = int(h * FACE_MARGIN)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(frame.shape[1], x + w + margin_x)
    y2 = min(frame.shape[0], y + h + margin_y)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def choose_result(prediction):
    sorted_indices = np.argsort(prediction)[::-1]
    top_index = int(sorted_indices[0])
    second_index = int(sorted_indices[1]) if len(sorted_indices) > 1 else top_index

    top_score = float(prediction[top_index])
    second_score = float(prediction[second_index])
    show_name = top_score >= MIN_CONFIDENCE and (top_score - second_score) >= MIN_MARGIN

    if show_name:
        return display_names[top_index], top_score, top_index, second_index

    return "No clear match", top_score, top_index, second_index


def main():
    print(f"Using model: {MODEL_PATH.name}")
    print(f"Using labels: {LABELS_PATH.name}")
    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        if not ret or frame is None:
            continue

        face_region, face_box = extract_face_region(frame)
        prediction_source = face_region if face_region is not None else frame
        prediction_input = preprocess_frame(prediction_source)
        prediction = model.predict(prediction_input, verbose=0)[0]
        result_name, confidence_score, top_index, second_index = choose_result(prediction)

        display_frame = frame.copy()
        if face_box is not None:
            x1, y1, x2, y2 = face_box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            display_frame,
            f"{result_name}: {confidence_score * 100:.1f}%",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        for row, (label, score) in enumerate(zip(display_names, prediction), start=1):
            cv2.putText(
                display_frame,
                f"{label}: {score * 100:.1f}%",
                (10, 22 + (row * 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow("Webcam Image", display_frame)

        print(
            f"\rTop: {display_names[top_index]} {prediction[top_index] * 100:.1f}% | "
            f"Next: {display_names[second_index]} {prediction[second_index] * 100:.1f}%   ",
            end="",
            flush=True,
        )

        keyboard_input = cv2.waitKey(1)
        if keyboard_input == 27:
            break

    camera.release()
    cv2.destroyAllWindows()
    print()


if __name__ == "__main__":
    main()
