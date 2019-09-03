import numpy as np
from cv2 import imread
from os import walk
import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SIZE = 1
FONT_COLOR = (255, 255, 255)
FONT_THICKNESS = 2

PATH_TO_FOLDER = r'..\resources\nut_imgs'
RESIZE_PERCENT = 30


def resize_image(initial_image, resize_percent):
    new_width = int(initial_image.shape[1] * resize_percent / 100)
    new_height = int(initial_image.shape[0] * resize_percent / 100)
    new_dim = (new_width, new_height)

    # resize image
    resized_image = cv2.resize(initial_image, new_dim, interpolation=cv2.INTER_AREA)

    # print('Original Dimensions : ', initial_image.shape)
    # print('Resized Dimensions : ', resized_image.shape)

    return resized_image


def get_images_list(path_to_folder):
    images_list = []
    for root, dirs, files in walk(path_to_folder):
        for _file in files:
            path_to_file = path_to_folder + '\\' + str(_file)
            image = imread(path_to_file)
            images_list.append(image)
    return images_list


def make_resized_images_list(initial_images_list, resize_percent):
    resized_images_list = []
    for initial_image in initial_images_list:
        resized_image = resize_image(initial_image=initial_image, resize_percent=resize_percent)
        resized_images_list.append(resized_image)
    return resized_images_list


def show_images_list(images_list, image_title):
    count = 1
    for image in images_list:
        image = cv2.putText(image,
                            'image ' + str(count),
                            (10, 50),
                            FONT,
                            FONT_SIZE,
                            FONT_COLOR,
                            FONT_THICKNESS,
                            cv2.LINE_AA)

        cv2.imshow(image_title, image)
        count += 1

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    initial_images_list = get_images_list(PATH_TO_FOLDER)
    resized_images_list = make_resized_images_list(initial_images_list, resize_percent=RESIZE_PERCENT)
    show_images_list(resized_images_list, image_title='resized image')


if __name__ == '__main__':
    main()
