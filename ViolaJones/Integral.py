import numpy as np

def integral_image(image):
    """
        Finds an integral image for provided image.
            Sums all pixels above (x, y) and from original array(x, y) and stors in ndarray s(x, y)
            Take the pixel on the left and add to s(x, y) value

    :param image: ndarray image
    :return: integral image
    """
    ii = np.zeros(image.shape)
    s = np.zeros(image.shape)
    for x in range(len(image)):
        for y in range(len(image[x])):
            # from vj algorithm paper
            s[x][y] = s[x-1][y] + image[x][y] if x-1 >= 0 else image[x][y] # check up and add to img value
            ii[x][y] = ii[x][y-1]+s[x][y] if y-1 >= 0 else s[x][y] # check left and add to s value
    return ii

def powerfulIntegral(images, val):
    """
        Images are already in ndarray type; convert to integral image and attach value passed to it indication
    if the image is positive or negative.

    :param images: array of tuplets (name, ndarray of image)
    :param val: 1 or 0 for positive or negative image type
    :return: array o tuplets (name, integral image, value)
    """
    arr = []
    for img in images:
        int_img = integral_image(img)
        arr.append((int_img, val))
    return arr