import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        # using mediapipe to process the video
        self.face_detect = mp.solutions.face_detection
        self.draw = mp.solutions.drawing_utils
        self.detected_faces = self.face_detect.FaceDetection(minDetectionCon)

    '"function to detect faces and return the values of the bounding box"'
    def findFaces(self, img, draw=True):
        # converting the image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.detected_faces.process(img_rgb)
        #print(self.results)
        bounds  = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bounding_box = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bounds_x = int(bounding_box.xmin * iw), int(bounding_box.ymin * ih), \
                         int(bounding_box.width * iw), int(bounding_box.height * ih)
                bounds.append([id, bounds_x, detection.score])

                cv2.rectangle(img, bounds_x, (255, 0, 255), 2)
                cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bounds_x[0], bounds_x[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                            2, (0, 255, 0), 2)
        return img, bounds

def main ():
    vc = cv2.VideoCapture(0)#using camera
    #switch the line if you want to detect images in a video
    # vc = cv2.VideoCapture("myVideos/Faces.mp4")# change to the directory where your video is stored
    f_time = 0
    detector = FaceDetector(0.5)
    while True:
        success, img = vc.read()  # reading frames
        img, bounding_box = detector.findFaces(img)

        # reducing the framerate for the video to slow down
        time_t = time.time()
        fps = 1 / (time_t - f_time)
        time_t = f_time
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 2)
        cv2.imshow("image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()