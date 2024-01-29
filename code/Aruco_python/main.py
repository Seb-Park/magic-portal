from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import cv2
from PIL import Image
import numpy as np
from objloader import *
from imutils.video import VideoStream
from cv2 import aruco as aruco
import yaml
import imutils

"""
This is file loads and displays the 3d model on OpenGL screen.
"""
 
class OpenGLGlyphs:
  
    # constants
    INVERSE_MATRIX = np.array([[ 1.0, 1.0, 1.0, 1.0],
                               [-1.0,-1.0,-1.0,-1.0],
                               [-1.0,-1.0,-1.0,-1.0],
                               [ 1.0, 1.0, 1.0, 1.0]])

    VIEW_MATRIX = np.eye(4)
    view_init = False
    MARKER_SIZE = 0.43
    SCREEN_SIZE = 7
    smooth_f = 0.9
    marker_centers = np.array([[-1, -1], [-1, -1], [-1, -1], [-1, -1]])
    # lerped_marker_centers = marker_centers.copy()
 
    def __init__(self):
        # initialise webcam and start thread
        self.webcam = VideoStream(src=0).start()
        self.webcam = cv2.VideoCapture(0)
        # self.webcam = cv2.VideoCapture('https://10.38.2.222:8080/video')
        # self.webcam = VideoStream(src='https://172.18.130.226:8080/video').start()
        self.remote_cam = cv2.VideoCapture('https://10.38.2.222:8080/video')
        # self.webcam = VideoStream(src='https://172.18.130.226:8080/video').start()
        # self.webcam = cv2.VideoCapture(0)
        # self.webcam = 
 
        # initialise shapes
        self.hero = None
        self.file = None
        self.cnt = 1
 
        # initialise texture
        self.texture_background = None

        print("getting data from file")
        self.cam_matrix,self.dist_coefs,rvecs,tvecs = self.get_cam_matrix("calibration/camera_matrix_aruco.yaml")

    def get_cam_matrix(self,file):
        cam_matrix = np.array([[1.51363568e+03, 0.00000000e+00, 9.75121055e+02],
       [0.00000000e+00, 1.51363568e+03, 5.13627201e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        dist_coeff = np.array([[ 3.68628543e+00],
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
        rvecs = None
        tvecs = None
        # with open(file) as f:
        #     loader = yaml.Loader(f)
        #     loadeddict = yaml.load(f, loader)
        #     cam_matrix = np.array(loadeddict.get('camera_matrix'))
        #     dist_coeff = np.array(loadeddict.get('dist_coeff'))
        #     rvecs = np.array(loadeddict.get('rvecs'))
        #     tvecs = np.array(loadeddict.get('tvecs'))
        return cam_matrix,dist_coeff,rvecs,tvecs
 
    def _init_gl(self, Width, Height):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(37, 1.3, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
      
        glLightfv(GL_LIGHT0, GL_POSITION,  (-40, 300, 200, 0.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHTING)
        glEnable(GL_COLOR_MATERIAL)
         
        # Load 3d object
        File = 'Sinbad_4_000001.obj'
        self.hero = OBJ(File, swapyz=False)
 
        # assign texture
        glEnable(GL_TEXTURE_2D)
        self.texture_background = glGenTextures(1)
 
    def _draw_scene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
 
        # get image from webcam
        # image = self.webcam.read()
        _, image = self.webcam.read()
        res, image = self.remote_cam.read()
        # cv2.imshow("frame", frame)
        # image = imutils.resize(image,width=640)
 
        # convert image to OpenGL texture format
        bg_image = cv2.flip(image, 0)
        bg_image = np.zeros(bg_image.shape, dtype=np.uint8)
        bg_image = Image.fromarray(bg_image)     
        ix = bg_image.size[0]
        iy = bg_image.size[1]
        bg_image = bg_image.tobytes("raw", "BGRX", 0, -1)
  
        # create background texture
        glBindTexture(GL_TEXTURE_2D, self.texture_background)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_image)
         
        # draw background
        glBindTexture(GL_TEXTURE_2D, self.texture_background)
        glPushMatrix()
        glTranslatef(0.0,0.0,-10.0)
        self._draw_background()
        glPopMatrix()
 
        # handle glyphs
        # image = self._handle_glyphs(image)
        self._handle_glyphs(image)

        glutSwapBuffers()
        
        gl_w, gl_h = glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT)
        edited = np.zeros((gl_h, gl_w, 3), dtype=np.uint8)
        glReadPixels(0, 0, gl_w, gl_h, GL_BGR, GL_UNSIGNED_BYTE, edited)
        edited = np.flip(edited, axis=0)
        # edited = np.flip(edited, axis=2)
        if(self.view_init):
            # h, w = self.find_shape_embed((gl_h, gl_w), image.shape)
            new_from_points = self.scale_four_points((gl_h, gl_w), 
                                             (image.shape[0], image.shape[1]), 
                                             self.marker_centers)
            target_points = np.array([[0, gl_h], 
                                      [gl_w, gl_h], 
                                      [gl_w, 0], 
                                      [0, 0]])
            M = cv2.findHomography(new_from_points, target_points, cv2.USAC_MAGSAC)[0]
        # lined = edited.copy()
        # for i in range(-1, 3):
        #     lined = cv2.line(lined, 
        #                      new_from_points[i].astype(np.int32), 
        #                      new_from_points[i + 1].astype(np.int32), 
        #                      (0, 255, 0), 9) 
        # cv2.imshow("lined", lined)
        warped = cv2.warpPerspective(edited, M, (gl_w, gl_h))
        cv2.imshow("Warped", warped)
 
    def _handle_glyphs(self, image):
        # aruco data
        aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
        parameters =  aruco.DetectorParameters_create()

        height, width, channels = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if ids is not None and corners is not None and len(ids) == 4: 
            rmtx = None
            all_tvecs = []
            marker_center_temp = [None, None, None, None]
            for i, id in enumerate(ids):
                # rvecs, tvecs, _objpoints = aruco.estimatePoseSingleMarkers(corners[0], 0.6, self.cam_matrix,self.dist_coefs)
                rvecs, tvecs, _objpoints = aruco.estimatePoseSingleMarkers(corners[i], self.MARKER_SIZE, self.cam_matrix,self.dist_coefs)
                #build view matrix
                # board = aruco.GridBoard_create(6,8,0.05,0.01,aruco_dict)
                # corners, ids, rejectedImgPoints,rec_idx = aruco.refineDetectedMarkers(gray,board,corners,ids,rejectedImgPoints)
                # ret,rvecs,tvecs = aruco.estimatePoseBoard(corners,ids,board,self.cam_matrix,self.dist_coefs)
                if rmtx is None:
                    rmtx = cv2.Rodrigues(rvecs)[0]
                
                all_tvecs.append(tvecs[0][0])
                marker_center_temp[int(id)] = self.find_center_four_points(corners[i][0])

            avg_tvecs = np.mean(np.array(all_tvecs), axis=0)

            view_matrix = np.array([[rmtx[0][0],rmtx[0][1],rmtx[0][2],avg_tvecs[0]],
                                    [rmtx[1][0],rmtx[1][1],rmtx[1][2],avg_tvecs[1]],
                                    [rmtx[2][0],rmtx[2][1],rmtx[2][2],avg_tvecs[2]],
                                    [0.0       ,0.0       ,0.0       ,1.0    ]])

            # view_matrix = np.array([[rmtx[0][0],rmtx[0][1],rmtx[0][2],tvecs[0]],
            #                         [rmtx[1][0],rmtx[1][1],rmtx[1][2],tvecs[1]],
            #                         [rmtx[2][0],rmtx[2][1],rmtx[2][2],tvecs[2]],
            #                         [0.0       ,0.0       ,0.0       ,1.0    ]])

            view_matrix = view_matrix * self.INVERSE_MATRIX

            view_matrix = np.transpose(view_matrix)

            if(not self.view_init):
                self.VIEW_MATRIX = view_matrix
                self.marker_centers = np.array(marker_center_temp)
                self.view_init = True
            else:
                self.VIEW_MATRIX = self.VIEW_MATRIX * self.smooth_f + view_matrix * (1 - self.smooth_f)
                self.marker_centers = self.marker_centers * self.smooth_f + np.array(marker_center_temp) * (1 - self.smooth_f)

            # load view matrix and draw shape
        if self.view_init:
            glPushMatrix()
            glLoadMatrixd(self.VIEW_MATRIX)

            glCallList(self.hero.gl_list)

            glPopMatrix()

            # cv2.imshow("cv frame", np.flip(image, axis=2))
            # cv2.imshow("cv frame", image)

            cv2.waitKey(1)
        

    def _draw_background(self):
        # draw background
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex3f(-4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 1.0); glVertex3f( 4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 0.0); glVertex3f( 4.0,  3.0, 0.0)
        glTexCoord2f(0.0, 0.0); glVertex3f(-4.0,  3.0, 0.0)
        glEnd( )

    def find_shape_embed(self, shape1, shape2):
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
    
    def scale_four_points(self, shape1, shape2, pts):
        '''
        pts - points in shape 2 that are to be scaled appropriately
        to be in scale with shape 1
        '''
        # Scales shape 1 to exactly cover shape 2 and returns the new shape 1
        h_w_ratio_1 = float(shape1[0]) / float(shape1[1])
        h_w_ratio_2 = float(shape2[0]) / float(shape2[1])

        scale = 1

        ## shape 1 is wider in proportion than shape 2
        if h_w_ratio_1 < h_w_ratio_2:
            ## If so the height of the new shape is that of shape 2, but the width is scaled and wider
            scale = float(shape1[0]) / float(shape2[0])
        # elif h_w_ratio_1 > h_w_ratio_2:
        else:
            ## If so the width of the new shape is that of shape 2, but the height is scaled and taller
            scale = float(shape1[1]) / float(shape2[1])
        ## They are exactly the same
        # else:
        #     return shape2
        # return np.array([
        #         [pts[0][0], pts[0][1]],
        #         [pts[1][0], pts[1][1]],
        #         [pts[2][0], pts[2][1]],
        #         [pts[3][0], pts[3][1]]
        #     ])
        return np.array(pts) * scale

    def find_center_four_points(self, four_points):
        # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line
        c, b, d, a = four_points
        px_num = (((a[0]*b[1]-a[1]*b[0])*(c[0]-d[0]))-((a[0]-b[0])*(c[0]*d[1]-c[1]*d[0])))
        py_num = (((a[0]*b[1]-a[1]*b[0])*(c[1]-d[1]))-((a[1]-b[1])*(c[0]*d[1]-c[1]*d[0])))
        p_den = ((a[0]-b[0]) * (c[1]-d[1])) - ((a[1] - b[1]) * (c[0] - d[0]))
        return [px_num / p_den, py_num / p_den]
 
    def main(self):
        # setup and run OpenGL
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        screen_dims = (2560, 1600)
        screen_dims = (640, 480)
        glutInitWindowSize(*screen_dims)
        glutInitWindowPosition(500, 400)
        self.window_id = glutCreateWindow(b"OpenGL Glyphs")
        glutDisplayFunc(self._draw_scene)
        glutIdleFunc(self._draw_scene)
        self._init_gl(*screen_dims)
        glutMainLoop()
  
# run an instance of OpenGL Glyphs 
openGLGlyphs = OpenGLGlyphs()
openGLGlyphs.main()