import numpy as np
import cv2

def detect_markers(input, num_markers, marker_type):
    marker_ids = np.array([])
    markerCorners = np.array([])
    rejectedCandidates = np.array([])
    detectorParams = cv2.aruco.DetectorParameters()
    aruco_dict = cv2.aruco.getPredefinedDictionary(marker_type)
    detector = cv2.aruco.ArucoDetector(aruco_dict, detectorParams)
    markerCorners, marker_ids, rejectedCandidates = detector.detectMarkers(input, markerCorners, marker_ids, rejectedCandidates)
    outputImage = input.copy()
    cv2.aruco.drawDetectedMarkers(outputImage, markerCorners, marker_ids)
    return outputImage

cap = cv2.VideoCapture('https://192.168.7.243:8080/video')

while True: 
    ret, frame = cap.read()
    detected = detect_markers(frame, 4, cv2.aruco.DICT_5X5_100)
    cv2.imshow("frame", detected)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()