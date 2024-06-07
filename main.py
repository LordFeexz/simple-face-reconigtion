import cv2
from typing import Sequence


class App:
    def __init__(self) -> None:
        self.face_ref = cv2.CascadeClassifier('face_ref.xml')
        self.camera = cv2.VideoCapture(index=0)

    def face_detection(self, frame) -> Sequence:
        return self.face_ref.detectMultiScale(
            cv2.cvtColor(
                src=frame,
                code=cv2.COLOR_RGB2GRAY
            ),
            scaleFactor=1.1,
            minSize=(300, 300),
            minNeighbors=3
        )

    def drawer_box(self, frame):
        for x, y, w, h in self.face_detection(frame=frame):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 4)

    def start(self):
        while True:
            _, frame = self.camera.read()
            self.drawer_box(frame=frame)
            cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.on_close()

    def on_close(self):
        self.camera.release()
        cv2.destroyAllWindows()
        exit()


if __name__ == "__main__":
    app = App()
    app.start()
