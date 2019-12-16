#!/usr/bin/env python3

import argparse
import numpy as np
import scipy.optimize as opt
import tensorflow as tf
import tensorflow.keras as ke

from skimage.transform import rescale
from skimage.io import imread, imsave


def transfer(
        content_img,
        style_img,
        content_size=None,
        style_size=None,
        content_layer="block4_conv2",
        content_weight=1.0,
        style_weight=1.0,
        variation_weight=1.0,
        noise=0.0,
        n_iter=500):
    imagenet_mean = np.array([103.939, 116.779, 123.68], dtype="float32")

    def vgg_prep(img):
        img = img.astype("float32")
        img = np.flip(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        img = img + imagenet_mean
        return img

    def vgg_un_prep(img):
        img = img - imagenet_mean
        img = np.squeeze(img, axis=0)
        img = np.flip(img, axis=-1)
        img = img.astype("uint8")
        img = np.clip(img, 0, 255)
        return img

    def width_resize(img, size):
        if size is not None:
            if img.shape[0] > img.shape[1]:
                scale = size / img.shape[0]
            else:
                scale = size / img.shape[1]
            img = rescale(img, scale, preserve_range=False, multichannel=True)
            img = img * 255.0
        return img

    content_img = width_resize(content_img, content_size)
    content_img = vgg_prep(content_img)

    style_img = width_resize(style_img, style_size)
    style_img = vgg_prep(style_img)

    vgg19 = ke.applications.VGG19(include_top=False, weights="imagenet")

    content_output = vgg19.get_layer(content_layer).output
    content_model = ke.Model(vgg19.input, outputs=content_output)
    content_map = content_model(content_img)

    style_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1"]
    style_outputs = [vgg19.get_layer(name).output for name in style_layers]
    style_model = ke.Model(vgg19.input, outputs=style_outputs)
    style_maps = style_model(style_img)

    def gram_matrix(x):
        x = tf.reshape(x, (-1, x.shape[-1]))
        gram = tf.transpose(x) @ x / x.shape[0]
        return gram

    def content_loss(x):
        x_map = content_model(x)
        loss = content_weight * tf.reduce_mean((x_map - content_map) ** 2)
        return loss

    def style_loss(x):
        x_maps = style_model(x)
        loss = 0.0
        for i in range(len(x_maps)):
            gram_1 = gram_matrix(x_maps[i])
            gram_2 = gram_matrix(style_maps[i])
            loss += style_weight * tf.reduce_mean((gram_1 - gram_2) ** 2)
        return loss

    def variation_loss(x):
        loss = variation_weight * tf.reduce_sum(tf.image.total_variation(x))
        return loss

    last_loss = 0.0

    def combined_loss(x):
        x = tf.cast(x, "float32")
        x = tf.reshape(x, content_img.shape)

        with tf.GradientTape() as tape:
            tape.watch(x)
            loss = content_loss(x) + style_loss(x) + variation_loss(x)
        grad = tape.gradient(loss, x)

        loss = loss.numpy()
        nonlocal last_loss
        last_loss = loss
        grad = grad.numpy().flatten().astype("float64")
        return loss, grad

    k = 0

    def callback(_):
        nonlocal k
        k += 1
        n_digits = len(str(n_iter))
        format_str = "Iteration: {:>" + str(n_digits) + "d}/{:>" + str(n_digits) + "d}; Loss: {:.3e}\r"
        print(format_str.format(k, n_iter, last_loss), end="")

    l_bound = vgg_prep(np.zeros(content_img.shape))
    u_bound = vgg_prep(255.0 * np.ones(content_img.shape))

    start = content_img + np.random.uniform(-noise, noise)
    start = np.clip(start, l_bound, u_bound)

    result = opt.minimize(
        combined_loss,
        start,
        method="L-BFGS-B",
        jac=True,
        callback=callback,
        bounds=opt.Bounds(l_bound.flatten(), u_bound.flatten()),
        options={"maxiter": n_iter})
    print()
    print("Exit status: {}".format(result.message))

    out_img = result.x.reshape(content_img.shape)
    out_img = vgg_un_prep(out_img)
    return out_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "content_img", type=str,
        help="content image filename")
    parser.add_argument(
        "style_img", type=str,
        help="style image filename")
    parser.add_argument(
        "-c", "--content_size", default=None, type=int,
        help="if given, the longer side of the content image is resized to this size")
    parser.add_argument(
        "-s", "--style_size", default=None, type=int,
        help="if given, the longer side of the style image is resized to this size")
    parser.add_argument(
        "-l", "--content_layer", default="block4_conv2", type=str,
        help="name of the content layer")
    parser.add_argument(
        "--content_weight", default=1.0, type=float,
        help="weight for the content loss")
    parser.add_argument(
        "--style_weight", default=10.0, type=float,
        help="weight for the style loss from each style layer")
    parser.add_argument(
        "--variation_weight", default=0.01, type=float,
        help="amount to penalize variation (noise) in the output image")
    parser.add_argument(
        "-n", "--noise", default=0.0, type=float,
        help="amount of noise to add to the content image before style transfer (scale is 0-255)")
    parser.add_argument(
        "-i", "--n_iter", default=500, type=int,
        help="maximum number of iterations for L-BFGS optimizer")
    parser.add_argument(
        "-o", "--out_img", default="out.png", type=str,
        help="output filename")

    args = parser.parse_args()

    out = transfer(
        imread(args.content_img),
        imread(args.style_img),
        content_size=args.content_size,
        style_size=args.style_size,
        content_layer=args.content_layer,
        noise=args.noise,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        variation_weight=args.variation_weight,
        n_iter=args.n_iter)

    imsave(args.out_img, out)
