# import cv2

# # Ptr<aruco::CharucoBoard> board = ... ## create charuco board
# board = cv2.aruco.CharucoBoard()
# image_size = (None, None) ## camera image size
# # vector<vector<Point2f>> allCharucoCorners;
# all_charuco_corners = []
# # vector<vector<int>> allCharucoIds;
# all_charuco_ids = []
# # vector<vector<Point2f>> allImagePoints;
# all_image_points = []
# # vector<vector<Point3f>> allObjectPoints;
# all_object_points = []
# # // Detect charuco board from several viewpoints and fill
# # // allCharucoCorners, allCharucoIds, allImagePoints and allObjectPoints
# detector = cv2.aruco.CharucoDetector()

# while(inputVideo.grab()):
#     detector.detectBoard(
#        image,
#        curr_charuco_corners,
#        curr_charuco_ids,
#        curr_marker_corners,
#        curr_marker_ids
#     )
#      {
#     detector.detectBoard(
#         image, currentCharucoCorners, currentCharucoIds
#     );
#     board.matchImagePoints(
#         currentCharucoCorners, currentCharucoIds,
#         currentObjectPoints, currentImagePoints
#     );
#     ...
# }
# // After capturing in several viewpoints, start calibration
# Mat cameraMatrix, distCoeffs;
# vector<Mat> rvecs, tvecs;
# // Set calibration flags (same than in calibrateCamera() function)
# int calibrationFlags = ...
# double repError = calibrateCamera(
#     allObjectPoints, allImagePoints, imageSize,
#     cameraMatrix, distCoeffs, rvecs, tvecs, noArray(),
#     noArray(), noArray(), calibrationFlags
# );