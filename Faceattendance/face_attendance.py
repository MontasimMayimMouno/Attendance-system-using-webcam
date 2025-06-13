import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Load student images
path = 'students'
images = []
names = []

for filename in os.listdir(path):
    img = cv2.imread(f'{path}/{filename}')
    if img is not None:
        images.append(img)
        names.append(os.path.splitext(filename)[0])

# Encode known faces
def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(img)
        if enc:
            encode_list.append(enc[0])
    return encode_list

# Mark attendance to CSV
def mark_attendance(name):
    if not os.path.exists('attendance.csv'):
        with open('attendance.csv', 'w') as f:
            f.write('Name,Time\n')

    with open('attendance.csv', 'r+') as f:
        lines = f.readlines()
        recorded_names = [line.split(',')[0] for line in lines]
        if name not in recorded_names:
            now = datetime.now()
            time_str = now.strftime('%H:%M:%S')
            f.write(f'{name},{time_str}\n')

if __name__ == "__main__":
    print("Encoding started...")
    known_encodings = find_encodings(images)
    print("Encoding complete.")

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        faces_cur = face_recognition.face_locations(rgb_small)
        encodes_cur = face_recognition.face_encodings(rgb_small, faces_cur)

        for encode, loc in zip(encodes_cur, faces_cur):
            matches = face_recognition.compare_faces(known_encodings, encode)
            face_dist = face_recognition.face_distance(known_encodings, encode)
            match_index = np.argmin(face_dist)

            if matches[match_index]:
                name = names[match_index].upper()
                y1, x2, y2, x1 = loc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                mark_attendance(name)

        cv2.imshow('Attendance System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
