from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
from ursina.shaders import lit_with_shadows_shader
import cv2
from panda3d.core import PNMImage
from panda3d.core import GraphicsOutput
import numpy as np

app = Ursina()

random.seed(0)
Entity.default_shader = lit_with_shadows_shader

ground = Entity(model='plane', collider='box', scale=64, texture='grass', texture_scale=(4,4))

editor_camera = EditorCamera(enabled=False, ignore_paused=True)
player = FirstPersonController(model='cube', z=-10, color=color.orange, origin_y=-.5, speed=8, collider='box')
player.collider = BoxCollider(player, Vec3(0,1,0), Vec3(1,2,1))

gun = Entity(model='cube', parent=camera, position=(.5,-.25,.25), scale=(.3,.2,1), origin_z=-.5, color=color.red, on_cooldown=False)
gun.muzzle_flash = Entity(parent=gun, z=1, world_scale=.5, model='quad', color=color.yellow, enabled=False)

shootables_parent = Entity()
mouse.traverse_target = shootables_parent

def update():
    # print('hey')
    get_frame_as_np_array()
    # get_frame_2()
    pass

def draw():
    print('hi')

def pause_input(key):
    if key == 'tab':    # press tab to toggle edit/play mode
        editor_camera.enabled = not editor_camera.enabled

        player.visible_self = editor_camera.enabled
        player.cursor.enabled = not editor_camera.enabled
        gun.enabled = not editor_camera.enabled
        mouse.locked = not editor_camera.enabled
        editor_camera.position = player.position

        application.paused = editor_camera.enabled
    if key == 'space':
        # setattr(window, 'fullscreen', True)
        # get_frame_as_np_array()
        # get_frame_2()
        # print(len(dir(window)))
        # print(len(set(dir(window) + dir(app.win))))
        print(app.windowType)
        setattr(app, 'windowType', 'offscreen')
        # print(dir(app.win))
        print(app.windowType)
    if key == 'k':
        get_frame_2()


pause_handler = Entity(ignore_paused=True, input=pause_input)


def get_frame_as_np_array():
    # print(dir(app))
    # frame_texture = app.render
    # frame_pnm_image = PNMImage()
    # return frame_texture.node()
    # frame_texture.node().copyTo(frame_pnm_image)
    # frame_np_array = np.array(frame_pnm_image.get_data(), dtype=np.uint8)
    # return frame_np_array
    # Naive approach, slow
    # app.screenshot('../ursina_outputs/hello.png', 0)
    # la = cv2.imread('../ursina_outputs/hello.png')
    # cv2.imshow('capture', la)
    ###
    ###
    # print(dir(app.render))
    # print(app.render.getColor())
    # print(app.render.getChildren()[0])
    # print(dir(app.render.getChildren()[0]))
    ###
    # print(app.win.get_texture())
    frame_pnm_image = PNMImage()
    app.win.get_screenshot().store(frame_pnm_image)
    
    # print(frame_pnm_image)
    app.graphicsEngine.renderFrame()
    # frame_np_array = np.array(frame_pnm_image.get_data(), dtype=np.uint8)

def get_frame_2():
    ### https://stackoverflow.com/questions/56466623/panda3d-rendered-window-to-numpy-array
    app.graphicsEngine.renderFrame()
    dr = app.camNode.getDisplayRegion(0)
    tex = dr.getScreenshot()
    data = tex.getRamImage()
    v = memoryview(data).tolist()
    img = np.array(v,dtype=np.uint8)
    img = img.reshape((tex.getYSize(),tex.getXSize(),4))
    img = img[::-1]
    # cv2.imshow('img',img)
    # cv2.waitKey(0)

#### ENEMY STUFF
    
for i in range(16):
    Entity(model='cube', origin_y=-.5, scale=2, texture='brick', texture_scale=(1,2),
        x=random.uniform(-8,8),
        z=random.uniform(-8,8) + 8,
        collider='box',
        scale_y = random.uniform(2,3),
        color=color.hsv(0, 0, random.uniform(.9, 1))
        )

class Enemy(Entity):
    def __init__(self, **kwargs):
        super().__init__(parent=shootables_parent, model='cube', scale_y=2, origin_y=-.5, color=color.light_gray, collider='box', **kwargs)
        self.health_bar = Entity(parent=self, y=1.2, model='cube', color=color.red, world_scale=(1.5,.1,.1))
        self.max_hp = 100
        self.hp = self.max_hp

    def update(self):
        dist = distance_xz(player.position, self.position)
        if dist > 40:
            return

        self.health_bar.alpha = max(0, self.health_bar.alpha - time.dt)


        self.look_at_2d(player.position, 'y')
        hit_info = raycast(self.world_position + Vec3(0,1,0), self.forward, 30, ignore=(self,))
        # print(hit_info.entity)
        if hit_info.entity == player:
            if dist > 2:
                self.position += self.forward * time.dt * 5

    @property
    def hp(self):
        return self._hp

    @hp.setter
    def hp(self, value):
        self._hp = value
        if value <= 0:
            destroy(self)
            return

        self.health_bar.world_scale_x = self.hp / self.max_hp * 1.5
        self.health_bar.alpha = 1

# Enemy()
enemies = [Enemy(x=x*4) for x in range(4)]

sun = DirectionalLight()
sun.look_at(Vec3(1,-1,-1))
Sky()

app.run()