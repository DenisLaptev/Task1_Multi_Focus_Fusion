import pywt
import cv2
import numpy as np
from os import walk
from cv2 import imread

# threshold parameters
THRESHOLD = 70
MAX_VALUE_THRESHOLD = 255

PATH_TO_FOLDER = r'../resources/nut_imgs'
RESIZE_PERCENT = 100


def get_images_list(path_to_folder):
    images_list = []
    for root, dirs, files in walk(path_to_folder):
        for _file in files:
            path_to_file = path_to_folder + '/' + str(_file)
            image = imread(path_to_file)
            images_list.append(image)
    return images_list


def get_filtered_contours_list(contours):
    new_contours = []
    for cnt in contours:
        # area
        area = cv2.contourArea(cnt)
        if area > 3000:
            new_contours.append(cnt)

    return new_contours


def get_roi_list(images_list):
    roi_list = []
    count = 1
    for image in images_list:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, image_threshold = cv2.threshold(src=image_gray,
                                           thresh=70,
                                           maxval=255,
                                           type=cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(image_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        new_contours = get_filtered_contours_list(contours)

        x, y, w, h = cv2.boundingRect(new_contours[0])
        cv2.rectangle(image_gray, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi = image[y:y + h, x:x + h]
        roi_list.append(roi)

        cv2.drawContours(image_gray, new_contours, -1, (0, 255, 0), 2)

        title = "gray image " + str(count)
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, image_gray)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        count += 1

    return roi_list


def make_equal_size_image(initial_image, new_width, new_height):
    new_width = new_width * 1
    new_height = new_height * 1
    new_dim = (new_width, new_height)

    # resize image
    resized_image = cv2.resize(initial_image, new_dim, interpolation=cv2.INTER_AREA)

    return resized_image


def make_equal_size_image_list(initial_images_list, new_width, new_height):
    equal_size_image_list = []
    for initial_image in initial_images_list:
        resized_image = make_equal_size_image(initial_image, new_width, new_height)
        equal_size_image_list.append(resized_image)
    return equal_size_image_list


def make_result_roi_from_roi_list(roi_list):
    result_roi = roi_list[0]

    roi_width = roi_list[0].shape[0]
    roi_height = roi_list[0].shape[1]

    roi_list_b = []
    roi_list_g = []
    roi_list_r = []

    for roi in roi_list:
        b, g, r = cv2.split(roi)

        roi_list_b.append(b)
        roi_list_g.append(g)
        roi_list_r.append(r)

    for x in range(roi_width):
        for y in range(roi_height):

            av_px_b = 0
            for roi_b in roi_list_b:
                av_px_b += roi_b[y, x]
            av_px_b = int(av_px_b / len(roi_list_b))

            av_px_g = 0
            for roi_g in roi_list_g:
                av_px_g += roi_g[y, x]
            av_px_g = int(av_px_g / len(roi_list_g))

            av_px_r = 0
            for roi_r in roi_list_r:
                av_px_r += roi_r[y, x]
            av_px_r = int(av_px_r / len(roi_list_r))

            result_roi[y, x] = [av_px_b, av_px_g, av_px_r]

    cv2.namedWindow("result_roi", cv2.WINDOW_NORMAL)
    cv2.imshow('result_roi', result_roi)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result_roi


def make_result_image(roi_list):
    # take image1 as result image. We will put result_roi into this image.
    result_image = cv2.imread(PATH_TO_FOLDER + '/' + '001.jpg')

    # cvt to gray
    result_image_gray = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

    # threshold
    _, image_threshold = cv2.threshold(src=result_image_gray,
                                       thresh=THRESHOLD,
                                       maxval=MAX_VALUE_THRESHOLD,
                                       type=cv2.THRESH_BINARY)

    # find countours for result_image
    contours, hierarchy = cv2.findContours(image_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    new_contours = get_filtered_contours_list(contours)

    x, y, w, h = cv2.boundingRect(new_contours[0])
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    result_roi = make_result_roi_from_roi_list(roi_list)

    result_image[y:y + h, x:x + h] = result_roi

    return result_image


def main():
    # get initial images list
    initial_images_list = get_images_list(PATH_TO_FOLDER)

    # get roi list
    roi_list = get_roi_list(initial_images_list)

    # get EQUAL SIZED roi list
    equal_size_roi_list = make_equal_size_image_list(roi_list,
                                                     new_width=roi_list[0].shape[0],
                                                     new_height=roi_list[0].shape[1])
    # make result image
    result_image = make_result_image(roi_list=equal_size_roi_list)

    # show result image
    cv2.namedWindow("result_image", cv2.WINDOW_NORMAL)
    cv2.imshow('result_image', result_image)

    # save result image
    path_to_output_file = r'../output/result.png'
    cv2.imwrite(path_to_output_file, result_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
