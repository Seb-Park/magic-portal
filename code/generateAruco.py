import numpy as np
import matplotlib.pyplot as plt
import cv2

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# tag_dim = 7
# tag_size = 1000

def create_n_aruco_tags(num_tags, tag_dim=7, tag_size=1000, dir=None):
    '''
    Generates a specified number of aruco tags with a specified number of pixels
    and a specified tag resolution

    num_tags - the number of tags to generate
    tag_dim - the number of squares on the side length of the tag. Must be 4 - 7
    tag_size - the side length of the actual image. Must be 50, 100, 250, 1000
    dir (optional) - the directory to write images to

    For example, if I were to have a tag_dim of 7 and a tag_size of 1000 that
    would generate a tag with 49 squares of data but 1,000,000 pixels
    '''
    aruco_type = f"DICT_{tag_dim}X{tag_dim}_{tag_size}"
    arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
    res = []

    for id in range(num_tags):
        tag = cv2.aruco.generateImageMarker(arucoDict, id, tag_size)
        if dir is not None:
            filename = f"ARUCO_{tag_dim}x{tag_dim}_{tag_size}_{id}.jpg"
            cv2.imwrite(f"{dir}{filename}", tag)
        res.append(tag)

    return res

    # tag = np.zeros((tag_size, tag_size, 1), dtype="uint8")
    # cv2.aruco.drawDetectedMarkers(arucoDict, id, tag_size, tag, 1)

create_n_aruco_tags(4, tag_dim=5, tag_size=250, dir="../markers/")