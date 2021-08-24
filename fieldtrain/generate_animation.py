from os import listdir
from os.path import isfile, join

import imageio as imageio
from PIL import Image, ImageDraw

def generate_animation_from_folder(folder, duration=0.005, fps=10, frames=10000):

    filename = folder[:-1] + '.gif'
    onlyfiles = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    onlyfiles = sorted(onlyfiles[:frames])

    generate_animation(filename, onlyfiles, fps)

def generate_animation(filename, frames, fps):
    images = []

    print("read images.")
    for frame in frames:
        images.append(imageio.imread(frame))

    # Save into a GIF file that loops forever
    print("write animation.")
    # kargs = {'duration': 0.005}
    kargs = {'fps': fps, 'loop': 1}
    result = imageio.mimsave(filename, images, 'GIF', **kargs)
    # result = imageio.mimsave(filename, images, 'GIF')
    print("done.")

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


def main():

    path_root =  'images/'
    imagepath = 'aviatar-512-512-0.0001'
    path = path_root + imagepath + '/'

    print("read images from folder: ", path)
    generate_animation_from_folder(path, frames=150, fps=20)

if __name__ == "__main__":
    main()