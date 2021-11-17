import cv2
import mediapipe as mp
import time

# vc = cv2.VideoCapture("myVideos/Faces.mp4")
vc = cv2.VideoCapture(0)#using camera
f_time = 0

#using mediapipe to process the video
face_detect = mp.solutions.face_detection
draw = mp.solutions.drawing_utils
detected_faces = face_detect.FaceDetection()

while True:
    success, img = vc.read()#reading frames

    #converting the image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detected_faces.process(img_rgb)
    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            print(id, detection)
            bounding_box = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bounds = int(bounding_box.xmin * iw), int(bounding_box.ymin * ih),\
                     int(bounding_box.width * iw), int(bounding_box.height * ih)
            cv2.rectangle(img, bounds, (255, 0, 255), 2)

    #reducing the framerate for the video to slow down
    time_t = time.time()
    fps = 1/(time_t - f_time)
    time_t = f_time
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 2)
    cv2.imshow("image", img)
    cv2.waitKey(1)