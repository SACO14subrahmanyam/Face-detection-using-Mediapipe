import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.5)

while 1:
    success, img = cap.read()
    # in this it open the camera
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #change the color of the image it turns the image into blue,green and red to red green and blue
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            #for loop that iterates over the list of results.detections
            bboxC = detection.location_data.relative_bounding_box
            #bboxC is a variable that is used to store the relative bounding box information for each detected face.
            mpDraw.draw_detection(img, detection)
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            #these are the length and width and height of the box
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            #In the above it will draw the box detected image and (255,0,255)

    cv2.imshow("Image", img)
    # it is displaying the images on monitor or screen
    cv2.waitKey(1)




