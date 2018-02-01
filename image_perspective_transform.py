import numpy
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

# how to add dots
# plt.plot(image[1][0][0], image[1][0][1], '.')  # right top
# plt.plot(image[1][1][0], image[1][1][1], '.')  # right down
# plt.plot(image[1][2][0], image[1][2][1], '.')  # left down
# plt.plot(image[1][3][0], image[1][3][1], '.')  # left top


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

images.append(
    ("/home/afomin/projects/mj/image_perspective_transform/images/test_image_2.png",
        [
            [180, 44],  # right top dot
            [144, 152],  # right down dot
            [16, 115],  # left down dot
            [59, 10]]))  # left top dot
print(images)

for image in images:
    print(image)
    img = mpimg.imread(image[0])
    show_image(img)
    img_warped = warp(img, image[1])
    show_image(img_warped)
