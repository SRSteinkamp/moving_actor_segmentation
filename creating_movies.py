import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from creating_objects import BasicObject

class BasicMovieMaker:

    def __init__(self, image_shape=(224, 224), speed=(15, 25), size=(15,25)):
        self.image_shape = image_shape
        self.speed = speed
        self.size = size

    def make_objects(self, no_objects):
        self.no_objects = no_objects
        self.target_idx = np.random.randint(no_objects)

        objects = []
        for ii in range(no_objects):

            pos_x, pos_y = (np.random.randint(0, self.image_shape[0]),
                            np.random.randint(0, self.image_shape[1]))
            obj_speed = np.random.randint(self.speed[0], self.speed[1])

            size1, size2 = (np.random.randint(self.size[0], self.size[1]),
                            np.random.randint(self.size[0], self.size[1]))

            SO = BasicObject([pos_y, pos_x], (size1, size2), img_shape=self.image_shape,
                              speed=obj_speed, direction=np.random.choice([-1, 1], 2))

            objects.append(SO)

        self.objects = objects

    def draw_one_frame(self):
        bg = np.zeros(self.image_shape)

        for n, obj in enumerate(self.objects):
            bg, _ = obj.draw(bg, istarget=False)

            if n == self.target_idx:
                mask = np.zeros(self.image_shape)
                masks, marker = obj.draw(mask, istarget=True)
                masks = masks > 0

            obj.get_a_move_on()

        return bg, marker, masks

    def make_movie(self, no_frames):

        movie = np.empty((no_frames, *self.image_shape))
        target = np.empty((no_frames, *self.image_shape))
        masks = np.empty((no_frames, *self.image_shape))

        for frm in range(no_frames):
            movie[frm, :, :], target[frm, :, :], masks[frm, :, :] = self.draw_one_frame()

        return movie, target, masks

    def play_movie(self, movie, target, masks, f_size=(10, 10)):
        try:
            Writer = animation.ImageMagickWriter
        except:
            Writer = animation.writers['html']
        writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)

        def animate(i):
            plt.imshow(movie[i] + target[i] + masks[i])

        fig = plt.figure(figsize=f_size)
        ani = animation.FuncAnimation(fig, animate, frames=movie.shape[0], repeat=True)

        return fig, ani
