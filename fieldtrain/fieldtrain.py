import os
import random

from os import listdir
from os.path import isfile, join

# importing image object from PIL
import math

import imageio as imageio
from PIL import Image, ImageDraw

import gym
import numpy as np
import tensorflow.keras.models
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

import datetime


class Image_Net:

    def __init__(self, width, height):
        # tuning params
        self.width = width
        self.height = height
        self.batch_size = width * height
        self.replay_buffer = []

    def set_topology(self, dimensions=2, topology=(16, 16), color_channels=3, learning_rate = 0.0001):
        self.topology = topology
        self.dimensions = dimensions
        self.color_channels = color_channels
        self.lr = learning_rate

        self.main_network = self.build_network()

    def get_action_from_model(self, state):
        state = np.reshape(state, (1, self.dimensions))
        qvalues = self.main_network.predict(state)
        return np.argmax(qvalues[0])

    def build_network(self):
        model = Sequential()

        hidden_layer = 1

        for layer in self.topology:
            if hidden_layer==1:
                model.add(Dense(layer, input_dim=self.dimensions, activation='relu'))
            else:
                model.add(Dense(layer, activation='relu'))
        model.add(Dense(self.color_channels, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.lr))
        return model

    def predict(self, netin):
        # netin = [(x, y)]
        p = self.main_network.predict(netin)
        return p

    def add_training(self, x, y, r, g, b):
        self.replay_buffer.append((x, y, r, g, b))


    def train(self, epochs):
        x_train = np.zeros(shape=(self.batch_size, self.dimensions))
        y_train = np.zeros(shape=(self.batch_size, self.color_channels))

        sample = 0

        for (x, y, r, g, b) in self.replay_buffer:
            coordinate = [(x, y)]
            coordinate = np.reshape(coordinate, (1, self.dimensions))

            color = [(r, g, b)]
            color = np.reshape(color, (1, self.color_channels))

            x_train[sample] = coordinate
            y_train[sample] = color

            sample += 1

        x_train.reshape(self.batch_size, self.dimensions)
        y_train.reshape(self.batch_size, self.color_channels)

        history = self.main_network.fit(x_train, y_train, epochs=1, verbose=0)
        # history.history


    def load(self, filename, episode=-1):
        path = filename + (('.' + str(episode)) if episode >= 0 else '')
        self.main_network = tensorflow.keras.models.load_model(path)

    def save(self, filename, episode=-1):
        path = filename + '.' + str(episode)
        self.main_network.save(path, overwrite=True)


def generate_animation_from_folder(folder):

    filename = folder[:-1] + '.gif'
    onlyfiles = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    generate_animation(filename, onlyfiles)

def generate_animation(filename, frames):
    images = []

    print("read images.")
    for frame in frames:
        images.append(imageio.imread(frame))

    # Save into a GIF file that loops forever
    print("write animation.")
    kargs = {'duration': 0.005}
    result = imageio.mimsave(filename, images, 'GIF', **kargs)
    # result = imageio.mimsave(filename, images, 'GIF')
    print("done.")

def generate_chess_board(tiles, factor, path):

    dim = tiles * factor

    # creating new Image object
    img = Image.new("RGB", (dim, dim))

    # create rectangle image
    canvas = ImageDraw.Draw(img)

    for xt in range(tiles):
        for yt in range(tiles):
            color = get_color_chess(xt, yt)
            draw_tile(canvas, xt, yt, factor, color)

    img.save(path)

    return path

def save_image_matrix(matrix, path):

    w = matrix.shape[1]
    h = matrix.shape[0]

    # creating new Image object
    img = Image.new("RGB", (w, h))
    canvas = ImageDraw.Draw(img)

    for yt in range(h):
        for xt in range(w):
            color = matrix[yt][xt]
            draw_tile(canvas, xt, yt, 1, color)

    img.save(path)

    return path


def generate_model(image_path, dir, tag, topology, steps=1500, learning_rate=0.0001, scale=1):

    image = imageio.imread(image_path)

    save_image_matrix(image, 'images/temp.png')

    width = image.shape[1]
    height = image.shape[0]

    model = Image_Net(width, height)
    model.set_topology(dimensions=2, topology=topology, color_channels=3, learning_rate=learning_rate)


    for xt in range(width):
        for yt in range(height):
            if image.shape[2] == 3:
                r, g, b = image[yt][xt]  # get_color_chess(xt, yt)
            else:
                r, g, b, a = image[yt][xt]  # get_color_chess(xt, yt)

            xc = xt - width / 2
            yc = yt - height / 2
            model.add_training(xc, yc, r, g, b)

    model_path = tag + "-"
    for layer in topology:
        model_path += str(layer)
        model_path += "-"
    model_path += str(learning_rate)

    path_root = dir + model_path + '/'

    if not os.path.exists(path_root):
        os.makedirs(path_root)
        print("Directory ", path_root, " Created ")

    frames = []
    pad = "0000"
    for i in range(1500):
        model.train(1)
        img = generate_image(model, scale)


        str_num = pad[:-len(str(i))] + str(i)

        path_pattern = 'image.{}.png'
        image_name = path_pattern.format(str_num)
        path = path_root + image_name

        img.save(path)
        frames.append(path)

        print("Image: ", image_name)

    filename = path_root[:-1] + '.gif'

    generate_animation(filename, frames)

    return model


def generate_image(model, scale):

    width = model.width
    height = model.height

    predict_buffer = []

    for xt in range(width):
        for yt in range(height):
            x = xt - width / 2
            y = yt - height / 2
            predict_buffer.append((x, y))

    color_fields = model.predict(predict_buffer)

    img = draw_tiles(width, height, scale, color_fields)

    return img


def get_color_chess(x, y):

    if (x + y) % 2 == 0:
        r = 255
        g = 255
        b = 255
    else:
        r = 0
        g = 0
        b = 0

    return r, g, b


def draw_tile(canvas, xt, yt, scale, color):
    x = xt * scale
    y = yt * scale

    shape = [(x, y), (x + scale, y + scale)]
    canvas.rectangle(shape, fill=(color[0], color[1], color[2]))

def draw_tiles(width, height, scale, color_fields):

    # creating new Image object
    img = Image.new("RGB", (width*scale, height*scale))
    canvas = ImageDraw.Draw(img)

    for xt in range(width):
        for yt in range(height):
            r,g,b = color_fields[yt + xt * height]
            color = ( int(r),int(g),int(b))
            draw_tile(canvas, xt, yt, scale, color)

    return img

# WIDTH = 400
# HEIGHT = 400
# TILES = 5
# RES = 10

def main():


    imagepath = 'images/chess_5_5.png'
    tag = "chess55"
    # generate_chess_board(5, 10, imagepath)

    imagepath = 'images/lh_logo.png'
    tag = "lh"


    # imagepath = 'images/dreieck.png'
    # tag = "dreieck"

    # imagepath = 'images/g.png'
    # tag = "g"

    imagepath = 'images/AVIATAR_LOGO.jpg'
    tag = "aviatar"

    topology = (128, 128)
    generate_model(imagepath, 'images/', tag, topology, scale=1, steps=1500, learning_rate=0.0001)
    # generate_animation_from_folder('images/32-32-/')

if __name__ == "__main__":
    main()