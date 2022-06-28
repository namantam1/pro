# %%
import os
import cv2
import numpy as np
# import matplotlib.pyplot as plt
from deepface.detectors import OpenCvWrapper, SsdWrapper
from deepface.basemodels import Facenet, ArcFace

file_name = "encodings/database.npz"
changed= False


detector_model = OpenCvWrapper.build_model()
detector = OpenCvWrapper

input_shape = (160, 160)
model = Facenet.loadModel()

try:
    known_face_encodings, know_face_labels = np.load(file_name).values()
except IOError:
    known_face_encodings, know_face_labels = np.array([]), np.array([], str)
    changed = True


def save_data():
    np.savez(file_name, known_face_encodings, know_face_labels)

def distance(encodings, encoding):
    if len(encodings) == 0:
        return np.empty(0)

    return np.linalg.norm(encodings - encoding, axis=1)

def preprocess(image, shape, normalize=True):
    image = cv2.resize(image, shape)
    if normalize:
        image = image / 255

    return image


def add_image(path):
    global know_face_labels, known_face_encodings, changed

    root, _ = os.path.splitext(path)
    label = os.path.split(root)[-1]

    if not np.isin(label, know_face_labels):
        print(f"Adding {label}...")
        image = cv2.imread(path)
        faces = detector.detect_face(detector_model, image)
        
        print(f"{len(faces)} faces detected")
        if len(faces) == 0:
            return
        
        face, _ = faces[0]
        face = preprocess(face, input_shape)
        
        # plt.imshow(face)
        # plt.show()
        
        data = np.array([face])
        encoded = model.predict(data)[0]

        if know_face_labels.size == 0:
            known_face_encodings = np.array([encoded])
            know_face_labels = np.array([label])
        else:
            known_face_encodings = np.vstack([known_face_encodings, encoded])
            know_face_labels = np.append(know_face_labels, label)
        print(f"Added {label}")
        changed = True
    else:
        print(f"`{label}` already exists")    

for dir, _, files in os.walk("images"):
    for file in files:
        add_image(os.path.join(dir, file))

if changed:
    save_data()


# %%
video_cap = cv2.VideoCapture(0)

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    _, frame = video_cap.read()
    # break

    if process_this_frame:
        faces = detector.detect_face(detector_model, frame.copy())

        resize_faces = []
        for face, location in faces:
            face_locations.append(location)
            image = preprocess(face, input_shape)
            resize_faces.append(image)
        resize_faces = np.array(resize_faces)

        try:
            face_encodings = model.predict(resize_faces)
            print("predicted", len(face_encodings))
        except Exception:
            face_encodings = []

        faces_names = []
        for encoding in face_encodings:
            dis = distance(known_face_encodings, encoding)
            best_index = np.argmin(dis)

            if dis[best_index] < 0.6:
                name = f"{know_face_labels[best_index]}".capitalize()
            face_names.append(name)

    process_this_frame = not process_this_frame


    for (x, y, w, h), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (67, 67, 67), 1)
        cv2.putText(frame, name, (int(x+w/4),int(y+h/1.5)), 
            cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()