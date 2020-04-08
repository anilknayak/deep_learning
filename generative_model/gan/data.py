import matplotlib.pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np
import os

class Data:
    def __init__(self, dir_name, batch_size):
        self.dir_name = dir_name
        self.batch_size = batch_size
        self.all_images = []
        self.image_shape = (32, 32)
        self.train = []
        self.curr_train_batch = 0
        self.evaluate = []
        self.curr_eval_batch = 0

    def load_images_from_dir(self):
        valid_images = [".jpeg",".jpg",".png"]
        for f in os.listdir(self.dir_name):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            self.all_images.append(os.path.join(self.dir_name, f))
        total_images = len(self.all_images)
        print("total_images : ", total_images)
        chunk = int(total_images * 0.8)
        self.train = self.all_images[0:chunk]
        self.evaluate = self.all_images[chunk:]
        print("train_images : ", len(self.train))
        print("evaluate_images : ", len(self.evaluate))


    def get_train_batch(self):
        filenames = self.train[self.curr_train_batch*self.batch_size:(self.curr_train_batch+1)*self.batch_size]
        self.curr_train_batch+=1
        num_images = len(filenames)
        if num_images == 0:
            self.curr_train_batch = 0
            return None, None, None, False
        real, fake = self._read_label(num_images)
        images = []
        for filename in filenames:
            img = self._read_image(filename)
            images.append(np.array(img))
        images = np.array(images)
        print(np.shape(images), np.shape(real), np.shape(fake))
        return images, real, fake, True

    def get_evaluation_batch(self):
        filenames = self.evaluate[self.curr_eval_batch*self.batch_size:(self.curr_eval_batch+1)*self.batch_size]
        self.curr_eval_batch+=1
        num_images = len(filenames)
        if num_images == 0:
            self.curr_eval_batch = 0
            return None, None, None, False
        real, fake = self._read_label(num_images)
        images = []
        for filename in filenames:
            img = self._read_image(filename)
            images.append(np.array(img))
        images = np.array(images)
        return images, real, fake, True

    def _read_image(self, filename):
        img = plt.imread(filename)
        img = resize(img, self.image_shape, anti_aliasing=True)
        # Normalize image
        img = (img/255)*2-1
        return img

    def denormalize(self, image):
        return np.uint8((image+1)/2*255)

    def _read_label(self, size):
        return np.ones([size, 1]), np.zeros([size, 1])
