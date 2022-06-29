import os
import cv2
import numpy as np
# import matplotlib.pyplot as plt
import OpenCvWrapper, SsdWrapper
import Facenet, ArcFace
import Dlib
from Threshold import findThreshold

file_name = "encodings/database.npz"
changed= False
align = True
metrix = "euclidean"
# metrix = "cosine"

detector_model = SsdWrapper.build_model()
detector = SsdWrapper

# model = Dlib.loadModel()
# input_shape = (150, 150)

model = ArcFace.loadModel()
input_shape = (112, 112)

# input_shape = (160, 160)
# model = Facenet.loadModel()

# threshold = findThreshold("Dlib", metrix)
threshold = findThreshold("ArcFace", metrix)
# threshold = findThreshold("Facenet", metrix)

try:
    known_face_encodings, know_face_labels = np.load(file_name).values()
except IOError:
    known_face_encodings, know_face_labels = np.array([]), np.array([], str)
    changed = True

def distance(encodings, encoding):
    if len(encodings) == 0:
        return np.empty(0)

    if metrix == "euclidean":
        return np.linalg.norm(encodings - encoding, axis=1)
    else:
        a1 = np.sum(np.multiply(encodings, encoding), axis=1)
        b1 = np.sum(np.multiply(encodings, encodings), axis=1)
        c1 = np.sum(np.multiply([encoding], [encoding]), axis=1)
        return (1 - (a1 / (b1**.5 * c1**.5)))

def save_data():
    np.savez(file_name, known_face_encodings, know_face_labels)

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
        faces = detector.detect_face(detector_model, image, align=align)
        
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


# def get_locations(img):
#     faces = detector.detect_face(detector_model, img, False)
#     locations = map(lambda x: x[1], faces)
#     return list(locations)

def encode(img):
    faces = detector.detect_face(detector_model, img, align=False)
    locations = map(lambda x: x[1], faces)
    print("Face detected", len(faces))

    encoded_faces = []
    for face, _ in faces:
        # face = img[y:y+h, x:x+w]
        face = preprocess(face, input_shape)
        encoded_faces.append(face)
    if len(encoded_faces) == 0:
        return [], []
    return model.predict(np.array(encoded_faces)), list(locations)

video_cap = cv2.VideoCapture(0)

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    _, frame = video_cap.read()
    # break

    if process_this_frame:
        # face_locations = get_locations(frame)
        face_encodings, face_locations = encode(frame.copy())

        face_names = []
        for face_encoding in face_encodings:
            # Or instead, use the known face with the smallest distance to the new face
            face_distances = distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            # if matches[best_match_index]:
            print(face_distances[best_match_index], threshold)
            if face_distances[best_match_index] < threshold:
                name = f"{know_face_labels[best_match_index]}".capitalize()
            else:
                name = "unknown"
            face_names.append(name)

    process_this_frame = not process_this_frame

    # print(face_locations, face_names)
    for (x, y, w, h), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (67, 67, 67), 1)
        cv2.putText(frame, name, (int(x+w/4),int(y)), 
            cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 255), 1)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()