import numpy as np
import cv2

def detect_markers(input, marker_type):
    outputImage = input.copy()
    cv2.aruco.drawDetectedMarkers(outputImage, markerCorners, marker_ids)
    return outputImage

def find_center_four_points(four_points):
    # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line
    c, b, d, a = four_points
    px_num = (((a[0]*b[1]-a[1]*b[0])*(c[0]-d[0]))-((a[0]-b[0])*(c[0]*d[1]-c[1]*d[0])))
    py_num = (((a[0]*b[1]-a[1]*b[0])*(c[1]-d[1]))-((a[1]-b[1])*(c[0]*d[1]-c[1]*d[0])))
    p_den = ((a[0]-b[0]) * (c[1]-d[1])) - ((a[1] - b[1]) * (c[0] - d[0]))
    return [px_num / p_den, py_num / p_den]

marker_type = cv2.aruco.DICT_5X5_100

cap = cv2.VideoCapture('https://192.168.7.243:8080/video')

marker_ids = np.array([])
markerCorners = np.array([])
rejectedCandidates = np.array([])
detectorParams = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(marker_type)
detector = cv2.aruco.ArucoDetector(aruco_dict, detectorParams)

marker_centers = np.array([[-1, -1], [-1, -1], [-1, -1], [-1, -1]])

M = np.eye(3, 3)

while True: 
    ret, frame = cap.read()
    # print(frame.shape)
    h, w, _ = frame.shape
    markerCorners, marker_ids, rejectedCandidates = \
        detector.detectMarkers(frame, markerCorners, marker_ids, rejectedCandidates)
    detected = frame.copy()
    cv2.aruco.drawDetectedMarkers(detected, markerCorners, marker_ids)
    for m, marker in enumerate(markerCorners):
        ### Populate marker data
        next_marker_index = m + 1 if m < len(markerCorners) - 1 else 0
        next_marker = markerCorners[next_marker_index]
        # # print("marker: ", marker)
        # for c, corner in enumerate(marker[0]):
        #     # print("corner: ", corner)
        #     # print(type(corner))
        #     detected = cv2.line(detected, corner.astype(np.int32), next_marker[0, c].astype(np.int32), (0, 0, 0), 6) 
        #     # detected = cv2.line(detected, corner, next_marker[c], (0, 0, 0), 6) 
        ct = np.array(find_center_four_points(marker[0]), dtype=np.int32)
        rect_radius = 60
        detected = cv2.rectangle(detected, ct - rect_radius, \
                                 ct + rect_radius, (0, 255, 255), 4) 
        # print(marker_ids[m])
        marker_centers[marker_ids[m][0]] = ct
    if -1 not in marker_centers: ## All values default to -1, if no -1, array has been populated
        # print("populated markers successfully")
        target_points = np.array([[0, h], [w, h], [w, 0], [0, 0]])
        M = cv2.findHomography(np.array(marker_centers), target_points, cv2.USAC_MAGSAC)[0]


    detected = cv2.warpPerspective(detected, M, (detected.shape[1], detected.shape[0]))
    cv2.imshow(f"frame", detected)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()