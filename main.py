import cv2
from typing import List
from os import path
import numpy as np
import face_recognition
import pandas as pd
from datetime import datetime


class App:
    def __init__(self) -> None:
        self.camera = cv2.VideoCapture(index=0)
        self.SAVED_FILE = 'known_faces.npy'
        self.known_faces_encodings, self.known_faces_names = self.load_known_face()
        self.df = pd.DataFrame(columns=["Name", "Time"])

    def save_known_faces(self):
        np.save(
            file=self.SAVED_FILE,
            arr={
                'encodings': self.known_faces_encodings, 'names': self.known_faces_names
            }
        )

    def add_new_faces(self, frame):
        name = input('Input Your Name: ')
        if name:
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(
                frame,
                face_locations
            )
            if face_encodings:
                self.known_faces_encodings.append(face_encodings[0])
                self.known_faces_names.append(name)
                self.save_known_faces()
                print(f"'{name}' face has been saved")

    def load_known_face(self) -> tuple[List, List]:
        if path.exists(self.SAVED_FILE):
            data = np.load(self.SAVED_FILE, allow_pickle=True).item()
            return data['encodings'], data['names']
        return [], []

    def save_data(self, name: str) -> None:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_data = pd.DataFrame(
            [[name, current_time]],
            columns=["Name", "Time"]
        )
        self.df = pd.concat([self.df, new_data], ignore_index=True)

    def drawer_box(self, frame: cv2.typing.MatLike) -> None:
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(
            frame,
            face_locations
        )

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(
                self.known_faces_encodings, face_encoding)
            name = "Unknown"
            color = (0, 0, 255)

            if self.known_faces_encodings:
                face_distances = face_recognition.face_distance(
                    self.known_faces_encodings,
                    face_encoding
                )
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_faces_names[best_match_index]
                    color = (255, 0, 0)

            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                color,
                2
            )
            cv2.rectangle(
                frame,
                (left, bottom - 35),
                (right, bottom),
                color,
                cv2.FILLED
            )
            cv2.putText(
                frame,
                name,
                (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255),
                1
            )
            self.save_data(name)

    def start(self):
        while True:
            ret, frame = self.camera.read()
            if not ret:
                break

            small_frame = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)

            self.drawer_box(frame=small_frame)
            cv2.imshow('Face Detection', small_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.on_close()
            elif key == ord('n'):
                self.add_new_faces(frame=small_frame)

    def on_close(self):
        self.camera.release()
        cv2.destroyAllWindows()
        exit()


if __name__ == "__main__":
    App().start()
