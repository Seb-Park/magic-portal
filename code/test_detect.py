import numpy as np
import cv2

def detect_markers(input, marker_type):
    # std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
    marker_ids = np.array([], dtype=np.int32)
    markerCorners = np.array([])
    rejectedCandidates = np.array([])
    detectorParams = cv2.aruco.DetectorParameters()
    aruco_dict = cv2.aruco.getPredefinedDictionary(marker_type)
    detector = cv2.aruco.ArucoDetector(aruco_dict, detectorParams)
    markerCorners, marker_ids, rejectedCandidates = detector.detectMarkers(input, markerCorners, marker_ids, rejectedCandidates)
    # # print(marker_corners)
    # if(len(marker_ids) + len(rejectedCandidates) + len(markerCorners) > 0):
    #     print("Marker Ids: ", marker_ids)
    #     print("Rejected: ", rejectedCandidates)
    #     print("Marker Corners: ", markerCorners)
    outputImage = input.copy()
    outputImage = cv2.aruco.drawDetectedMarkers(outputImage, markerCorners, marker_ids)
    return outputImage

my_img = cv2.imread('../IMG_2482.jpg')
detected = detect_markers(my_img, cv2.aruco.DICT_5X5_250)
cv2.imshow("frame", detected)
cv2.waitKey(0)
cv2.destroyAllWindows()