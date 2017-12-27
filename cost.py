import numpy as np
import tensorflow as tf


def randomInitiation(content, noise_ratio=0.6):
    content_noise = np.random.uniform(-20, 20, content.shape).astype('float32')
    return content_noise * noise_ratio + content * (1 - noise_ratio)


def _costFunctionContent(content, generation):
    J_content = tf.reduce_sum(tf.pow(content - generation, 2)) / (4.0 * content.shape[0] * content.shape[1] * content.shape[2])
    return J_content


def _costFuntionStyle(style, generation):
    tf.cast(style, tf.float32)
    tf.cast(generation, tf.float32)

    nh = style.shape[0]
    nw = style.shape[1]
    nc = style.shape[2]
    divisor = tf.pow(2.0 * nh * nw * nc, 2)

    style_reshaped = tf.reshape(style, (nh * nw, nc))
    generation_reshaped = tf.reshape(generation, (nh * nw, nc))
    A = tf.matmul(tf.transpose(style_reshaped), style_reshaped)
    G = tf.matmul(tf.transpose(generation_reshaped), generation_reshaped)
    J_Style = tf.reduce_sum(tf.pow(G - A, 2)) / divisor

    return J_Style


def costFunction(content, styles, weights, generation_content, generation_styles, alpha, betha):
    cost = alpha * _costFunctionContent(content, generation_content)
    for i in range(len(weights)):
        cost += betha * _costFuntionStyle(styles[i][0], generation_styles[i][0]) * weights[i]
    return cost
