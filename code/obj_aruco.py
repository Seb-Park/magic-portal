# Basic OBJ file viewer. needs objloader from:
#  http://www.pygame.org/wiki/OBJFileLoader
# LMB + move: rotate
# RMB + move: pan
# Scroll wheel: zoom in/out
import sys, pygame
from pygame.locals import *
from pygame.constants import *
from OpenGL.GL import *
from OpenGL.GLU import *

import cv2
import numpy as np

# IMPORT OBJECT LOADER
from objloader import *

pygame.init()
viewport = (800,600)
hx = viewport[0]/2
hy = viewport[1]/2
srf = pygame.display.set_mode(viewport, OPENGL | pygame.DOUBLEBUF)

glLightfv(GL_LIGHT0, GL_POSITION,  (-40, 200, 100, 0.0))
glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
glEnable(GL_LIGHT0)
glEnable(GL_LIGHTING)
glEnable(GL_COLOR_MATERIAL)
glEnable(GL_DEPTH_TEST)
glShadeModel(GL_SMOOTH)           # most obj files expect to be smooth-shaded

# LOAD OBJECT AFTER PYGAME INIT
obj = OBJ(sys.argv[1], swapyz=True, dir=sys.argv[2])

clock = pygame.time.Clock()

glMatrixMode(GL_PROJECTION)
glLoadIdentity()
width, height = viewport
gluPerspective(90.0, width/float(height), 1, 100.0)
glEnable(GL_DEPTH_TEST)
glMatrixMode(GL_MODELVIEW)

rx, ry = (0,0)
tx, ty = (0,0)
zpos = 5
rotate = move = False

marker_type = cv2.aruco.DICT_5X5_100

ip = '10.38.27.70:8080'
cap = cv2.VideoCapture(f'https://{ip}/video')
target_im = cv2.imread('../test.png')

marker_ids = np.array([])
marker_corners = np.array([])
rejected_candidates = np.array([])
detectorParams = cv2.aruco.DetectorParameters_create()
aruco_dict = cv2.aruco.getPredefinedDictionary(marker_type)

marker_centers = np.array([[-1, -1], [-1, -1], [-1, -1], [-1, -1]])
lerped_marker_centers = marker_centers.copy()
smooth_factor = 0.5

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

rvec, tvec = None, None

### https://stackoverflow.com/questions/61906181/how-to-get-frames-from-a-pygame-game
# # function that we can give two functions to and will return us a new function that calls both
# def function_combine(screen_update_func, our_intercepting_func):  
#     def wrap(*args, **kwargs):  
#         screen_update_func(*args,  
#                 **kwargs) # call the screen update func we intercepted so the screen buffer is updated  
#         our_intercepting_func() # call our own function to get the screen buffer  
#     return wrap  

# def on_screen_update():  
#     surface_array = pygame.surfarray.array3d(pygame.display.get_surface())  
#     print("We got the screen array")  
#     print(surface_array)  

#  # set our on_screen_update function to always get called whenever the screen updated  
# pygame.display.update = function_combine(pygame.display.update, on_screen_update)

while 1:
    ### VISION
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    marker_corners, marker_ids, rejected_candidates = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detectorParams)

    # marker_corners, marker_ids, rejected_candidates = cv2.aruco.detectMarkers(gray, aruco_dict, marker_corners, marker_ids, detectorParams, rejected_candidates)

    if len(marker_corners) > 0:
        for i, mid in enumerate(marker_ids):
            if(mid == 0):
                new_rvec, new_tvec, new_markerPoints = cv2.aruco.estimatePoseSingleMarkers(marker_corners[i], 0.02, mtx, dist)
                if new_rvec is not None:
                    if(rvec is None):
                        rvec = new_rvec
                        tvec = new_tvec
                    rvec = rvec * smooth_factor + (new_rvec * (1 - smooth_factor))
                    tvec = tvec * smooth_factor + (new_tvec * (1 - smooth_factor))
                # new = cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.02)
    
    # print(rvec)

    ### RENDERING

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # RENDER OBJECT
    
    # print(tx/20., ty/20., - zpos)
    if(rvec is not None):
        # print(*rvec[0][0])
        glTranslate(*(tvec[0][0] * [100, -100, -100]))
        rotate_mag = np.linalg.norm(rvec[0][0])
        glRotate(rotate_mag * 180 / np.pi, *(rvec[0][0][::-1]))
        # print(rotate_mag)
    else:
        glTranslate(tx/20., ty/20., - zpos)
    glRotate(ry, 1, 0, 0)
    glRotate(rx, 0, 1, 0)
    glCallList(obj.gl_list)

    pygame.display.flip()

    # surface = pygame.display.get_surface()
    # hello = pygame.surfarray.array3d(surface)
    # print(pygame.image.tobytes(pygame.display.get_surface(), "RGBA"))
    pygame.image.save(pygame.display.get_surface(), "../pygame_outputs/frame.png")

    srf.set_alpha(0)

    # cv2.imshow("Frame", np.array(pygame.surfarray.pixels2d(srf)))