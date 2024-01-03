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
target_im = cv2.imread('../test.png')

# target_im = target_im.resize((1800, 2880))

cv2.imshow(f"frame", target_im)
cv2.waitKey(0)

marker_ids = np.array([])
markerCorners = np.array([])
rejectedCandidates = np.array([])
detectorParams = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(marker_type)
detector = cv2.aruco.ArucoDetector(aruco_dict, detectorParams)

marker_centers = np.array([[-1, -1], [-1, -1], [-1, -1], [-1, -1]])

M = np.eye(3, 3)

def find_shape_embed(shape1, shape2):
    # Scales shape 1 to exactly cover shape 2 and returns the new shape 1
    h_w_ratio_1 = float(shape1[0]) / float(shape1[1])
    h_w_ratio_2 = float(shape2[0]) / float(shape2[1])

    ## shape 1 is wider in proportion than shape 2
    if h_w_ratio_1 < h_w_ratio_2:
        ## If so the height of the new shape is that of shape 2, but the width is scaled
        return (int(shape2[0]), int(shape2[0] / h_w_ratio_1))
    elif h_w_ratio_1 > h_w_ratio_2:
        ## If so the width of the new shape is that of shape 2, but the height is scaled
        return (int(shape2[1] * h_w_ratio_1), int(shape2[1]))
    ## They are exactly the same
    else:
        return shape2

while True: 
    ret, frame = cap.read()
    # print(frame.shape)
    h, w, _ = frame.shape
    h, w, _ = target_im.shape
    h, w = find_shape_embed(target_im.shape, frame.shape)
    markerCorners, marker_ids, rejectedCandidates = \
        detector.detectMarkers(frame, markerCorners, marker_ids, rejectedCandidates)
    detected = frame.copy()
    cv2.aruco.drawDetectedMarkers(detected, markerCorners, marker_ids)
    for m, marker in enumerate(markerCorners):
        ### Populate marker data
        next_marker_index = m + 1 if m < len(markerCorners) - 1 else 0
        next_marker = markerCorners[next_marker_index]
        ct = np.array(find_center_four_points(marker[0]), dtype=np.int32)
        rect_radius = 60
        detected = cv2.rectangle(detected, ct - rect_radius, \
                                 ct + rect_radius, (0, 255, 255), 4) 
        found_marker = marker_ids[m][0]
        if(found_marker < 4):
            ### Make sure if it thinks it found a marker it's one of the four
            marker_centers[found_marker] = ct
    if -1 not in marker_centers: ## All values default to -1, if no -1, array has been populated
        # print("populated markers successfully")
        target_points = np.array([[0, h], [w, h], [w, 0], [0, 0]])
        M = cv2.findHomography(np.array(marker_centers), target_points, cv2.USAC_MAGSAC)[0]

    detected = cv2.warpPerspective(target_im, M, (target_im.shape[1], target_im.shape[0]))
    cv2.imshow(f"frame", detected)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()