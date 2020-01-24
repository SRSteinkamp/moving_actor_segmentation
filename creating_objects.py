import numpy as np
from skimage import draw

class SimpleObject():

    def __init__(self, center, size, speed=2, mutation=1e-5, direction=[1, 1], img_shape=[256, 256]):
        self.center_x = center[0]
        self.center_y = center[1]
        self.size = np.array(size)
        self.speed = speed
        self.mutation = mutation
        self.img_shape = np.array(img_shape)
        self.direction = direction

    def draw(self, background, object_type='ellipse', istarget=False):

        assert np.all(background.shape == self.img_shape)

        self.simple_object = draw.ellipse(self.center_y, self.center_x, self.size[0],
                                          self.size[1], shape=self.img_shape)

        self.simple_obj_peri = draw.ellipse_perimeter(self.center_y, self.center_x, self.size[0],
                                          self.size[1], shape=self.img_shape)

        background[self.simple_object[0].astype(int), self.simple_object[1].astype(int)] = 1/2
        background[self.simple_obj_peri[0].astype(int), self.simple_obj_peri[1].astype(int)] = 1

        image = background

        if istarget:
            marker = np.zeros(self.img_shape)
            rm, rc = draw.rectangle((self.center_y - 2, self.center_x -2),
                                    (self.center_y + 2, self.center_x  + 2),
                                    shape=self.img_shape)
            marker[rm, rc] = 1
        else:
            marker = None
        return image, marker

    def get_a_move_on(self):

        step1 = np.random.randint(self.speed)
        step2 = (self.speed - step1) * self.direction[1]
        step1 = step1 * self.direction[0]
        if (self.center_x + step1) < 0 or (self.center_x + step1) >= (self.img_shape[1]):
            step1 = step1 * -1
            self.direction[0] = -1 * self.direction[0]

        if (self.center_y + step2) < 0 or (self.center_y + step2) >= (self.img_shape[0]):
            step2 = step2 * -1
            self.direction[1] = -1 * self.direction[1]

        self.center_x += step1
        self.center_y += step2
