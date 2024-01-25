import numpy as np
import cv2

# use older version of opencv

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

## Pose estimation
###https://stackoverflow.com/questions/75750177/solve-pnp-or-estimate-pose-single-markers-which-is-better

def my_estimatePoseSingleMarkers(corners, marker_width, marker_height, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_width / 2, marker_height / 2, 0],
                              [marker_width / 2, marker_height / 2, 0],
                              [marker_width / 2, -marker_height / 2, 0],
                              [-marker_width / 2, -marker_height / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

marker_type = cv2.aruco.DICT_5X5_100

ip = '10.38.27.70:8080'
cap = cv2.VideoCapture(f'https://{ip}/video')
target_im = cv2.imread('../test.png')

marker_ids = np.array([])
marker_corners = np.array([])
rejected_candidates = np.array([])
detectorParams = cv2.aruco.DetectorParameters_create()
aruco_dict = cv2.aruco.getPredefinedDictionary(marker_type)
# detector = cv2.aruco.ArucoDetector(aruco_dict, detectorParams)

marker_centers = np.array([[-1, -1], [-1, -1], [-1, -1], [-1, -1]])
lerped_marker_centers = marker_centers.copy()
smooth_factor = 0.5

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


mtx = np.array([[1.51363568e+03, 0.00000000e+00, 9.75121055e+02],
       [0.00000000e+00, 1.51363568e+03, 5.13627201e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([[ 3.68628543e+00],
       [-8.75820830e+01],
       [ 2.58461750e-03],
       [ 3.38325980e-03],
       [ 3.47102190e+02],
       [ 3.52159347e+00],
       [-8.61481611e+01],
       [ 3.42770774e+02],
       [ 0.00000000e+00],
       [ 0.00000000e+00],
       [ 0.00000000e+00],
       [ 0.00000000e+00],
       [ 0.00000000e+00],
       [ 0.00000000e+00]])

def augment(img, obj, projection, template, color=False, scale = 4):
    # takes the captureed image, object to augment, and transformation matrix  
    #adjust scale to make the object smaller or bigger, 4 works for the fox

    h, w = template.shape
    vertices = obj.vertices
    img = np.ascontiguousarray(img, dtype=np.uint8)

    #blacking out the aruco marker
    a = np.array([[0,0,0], [w, 0, 0],  [w,h,0],  [0, h, 0]], np.float64 )
    imgpts = np.int32(cv2.perspectiveTransform(a.reshape(-1, 1, 3), projection))
    cv2.fillConvexPoly(img, imgpts, (0,0,0))

    #projecting the faces to pixel coords and then drawing
    for face in obj.faces:
        #a face is a list [face_vertices, face_tex_coords, face_col]
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices]) #-1 because of the shifted numbering
        points = scale*points
        points = np.array([[p[2] + w/2, p[0] + h/2, p[1]] for p in points]) #shifted to centre 
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)#transforming to pixel coords
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (50, 50, 50))
        else:
            cv2.fillConvexPoly(img, imgpts, face[-1])
            
    return img

while True: 
    ret, frame = cap.read()
    # print(frame.shape)
    h, w, _ = frame.shape
    h, w, _ = target_im.shape
    h, w = find_shape_embed(target_im.shape, frame.shape)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    marker_corners, marker_ids, rejected_candidates = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detectorParams)

    # marker_corners, marker_ids, rejected_candidates = cv2.aruco.detectMarkers(gray, aruco_dict, marker_corners, marker_ids, detectorParams, rejected_candidates)
    
    if len(marker_corners) > 0:
        for i in range(0, len(marker_ids)):
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(marker_corners[i], 0.02, mtx, dist)
            new = cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.02)

    for m, marker in enumerate(marker_corners):
        ### Populate marker data
        ct = np.array(find_center_four_points(marker[0]), dtype=np.int32)
        rect_radius = 60
        detected = cv2.rectangle(detected, ct - rect_radius, \
                                 ct + rect_radius, (0, 255, 255), 4) 
        found_marker = marker_ids[m][0]
        if(found_marker < 4):
            ### Make sure if it thinks it found a marker it's one of the four
            marker_centers[found_marker] = ct
            if np.all(lerped_marker_centers[found_marker] >= 0):
                lerped_marker_centers[found_marker] = lerped_marker_centers[found_marker] * smooth_factor + (marker_centers[found_marker] * (1 - smooth_factor))
            else:
                lerped_marker_centers[found_marker] = ct
    if -1 not in marker_centers: ## All values default to -1, if no -1, array has been populated
        # print("populated markers successfully")
        target_points = np.array([[0, h], [w, h], [w, 0], [0, 0]])
        M = cv2.findHomography(np.array(lerped_marker_centers), target_points, cv2.USAC_MAGSAC)[0]
    
    cv2.imshow(f"frame", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()