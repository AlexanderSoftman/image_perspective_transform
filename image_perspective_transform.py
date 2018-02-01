import numpy
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

# how to add dots
# plt.plot(image[1][0][0], image[1][0][1], '.')  # right top
# plt.plot(image[1][1][0], image[1][1][1], '.')  # right down
# plt.plot(image[1][2][0], image[1][2][1], '.')  # left down
# plt.plot(image[1][3][0], image[1][3][1], '.')  # left top


def find_reference_dots(img):
    img_res = numpy.copy(img)
    # convert frame to gray
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # add gaussian smoothing to gray frame
    blur_gray_frame = gray_img
    # kernel_size = 5
    # blur_gray_frame = cv2.GaussianBlur(
    #     gray_img,
    #     (kernel_size, kernel_size),
    #     0)

    res = cv2.adaptiveThreshold(
        gray_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2)
    return (res)
    # ret, imgf = cv2.threshold(
    #     img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # define Canny parameters:
    low_threshold = 50
    high_threshold = 150
    canny_edges_from_gray_frame = cv2.Canny(
        blur_gray_frame,
        low_threshold,
        high_threshold)
    return (canny_edges_from_gray_frame)
    # define Hough transformation parameters
    rho = 1
    theta = numpy.pi / 180
    threshold = 1
    min_line_length = 500
    max_line_gap = 50

    # create empty image
    line_frame = numpy.copy(img) * 0
    lines = cv2.HoughLinesP(
        canny_edges_from_gray_frame,
        rho,
        theta,
        threshold,
        numpy.array([]),
        min_line_length,
        max_line_gap)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(
                line_frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                10)

    # # add lines to color_edges
    # frame_res = cv2.addWeighted(
    #     img,
    #     0.8,
    #     line_frame,
    #     1,
    #     0)
    return (line_frame)


def plot_hist(img_path):
    img = cv2.imread(img_path, 0)
    plt.subplot(2, 1, 1), plt.imshow(img, cmap='gray')
    plt.title('Original Noisy Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 1, 2), plt.hist(img.ravel(), 256)
    plt.title('Histogram'), plt.xticks([]), plt.yticks([])
    plt.show()


# function get img and returned warped img
def warp(img, polygon):
    img_size = (img.shape[1], img.shape[0])  # size format x, y

    # original point from image
    src = numpy.float32(polygon)

    # desired points
    dst = numpy.float32(
        [
            [img_size[0], 0],  # right top dot
            [img_size[0], img_size[1]],  # right down dot
            [0, img_size[1]],  # left down dot
            [0, 0]])  # left top dot

    # matrix of perspective transform
    M = cv2.getPerspectiveTransform(src, dst)

    # we can get inversed matrix
    # M_inv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped


def show_image(image):
    plt.imshow(image)
    plt.show()
    while(1):
        k = cv2.waitKey(0)
        if k:    # Esc key to stop
            break
    cv2.destroyAllWindows()


images = []
images.append(
    ("/home/afomin/projects/mj/image_perspective_transform/images/test_image_1.jpg",
        [
            [2456, 1706],  # right top dot
            [2455, 3327],  # right down dot
            [1171, 3465],  # left down dot
            [1087, 1738]]))  # left top dot

# images.append(
#     ("/home/afomin/projects/mj/image_perspective_transform/images/test_image_2.png",
#         [
#             [180, 44],  # right top dot
#             [144, 152],  # right down dot
#             [16, 115],  # left down dot
#             [59, 10]]))  # left top dot

for image in images:
    # plot_hist(image[0])
    img = mpimg.imread(image[0])
    show_image(img)
    # img_lines = find_reference_dots(img)
    # show_image(img_lines)
    img_warped = warp(img, image[1])
    show_image(img_warped)
