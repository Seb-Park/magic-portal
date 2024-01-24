import pygame as pg
from OpenGL.GL import *
import numpy as np
import ctypes
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr

class Mesh:
    def __init__(self, filepath):
        self.vertices = self.loadMesh(filepath)

        self.vertex_count = len(self.vertices) // 5

        
