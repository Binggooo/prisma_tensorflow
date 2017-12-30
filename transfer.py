import vgg16
import cost
import numpy as np
import tensorflow as tf
from PIL import Image

WEIGHT_STYLE = [0.5, 1.0, 1.5, 3.0, 4.0]
ALPHA = 0.1
BETHA = 0.001
LEARNING_RATE = 2.0
ITERATIONS = 101
FREQUENCY_SAVE = 10

# build vggNet
vgg = vgg16.Vgg16('./vgg16.npy')
print('vggNet is built...')

# load image
contentImage = np.array(Image.open('./examples/content.jpg')).reshape((1, vgg.HEIGHT, vgg.WIDTH, vgg.CHANNELS))
styleImage = np.array(Image.open('./examples/style.jpg')).reshape((1, vgg.HEIGHT, vgg.WIDTH, vgg.CHANNELS))
generation = cost.randomInitiation(contentImage)
print('image is loaded...')

with tf.Session() as sess:
    # feature maps
    sess.run(tf.global_variables_initializer())
    sess.run(vgg.inputRGB.assign(contentImage))
    content = sess.run(vgg.contentRGB)
    sess.run(vgg.inputRGB.assign(styleImage))
    styles = sess.run(vgg.stylesRGB)
    print('feature maps Get...')

    # graph
    total_cost = cost.costFunction(content, styles, WEIGHT_STYLE, vgg.contentRGB, vgg.stylesRGB, ALPHA, BETHA)
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    train = optimizer.minimize(total_cost)
    print('graph Get...')

    # train
    sess.run(tf.global_variables_initializer())
    sess.run(vgg.inputRGB.assign(generation))
    for i in range(ITERATIONS):
        print('iterations:', i, ' ,total_cost:', sess.run(total_cost))
        sess.run(train)
        if i % FREQUENCY_SAVE == 0:
            outputRGB = sess.run(vgg.outputRGB)[0]
            outputImage = Image.fromarray(np.clip(outputRGB, 0, 255).astype('uint8'))
            outputImage.save('./examples/pic%d.jpg' % i, mode='RGB')
