from deep_learning.generative_model.gan import gan_model, data
import matplotlib.image as mpimg

epochs = 2
batch_size = 2
dataset = data.Data("/Users/anilnayak/Desktop/dataset", batch_size)
dataset.load_images_from_dir()

gan = gan_model.GanModel(batch_size)
gan.network()
for epoch in range(epochs):
    print("Training Epoch: ", epoch)
    while True:
        real_images, y_real, y_fake, flag = dataset.get_train_batch()
        if not flag:
            break
        gan.train(epoch, real_images, y_real, y_fake)

    while True:
        real_images, y_real, y_fake, flag = dataset.get_evaluation_batch()
        if not flag:
            break
        gan.evaluate(epoch, real_images, y_real, y_fake)

gan.show_losses()

import numpy as np
images = gan.get_image(1)
for image in images:
    print(np.shape(image))
    image = dataset.denormalize(image)
    mpimg.imsave("generate_img.jpg", image)