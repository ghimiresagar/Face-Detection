import cv2
import os

"""
The script takes images from a path, resizes to 100x100, turns to grayscale
"""

def load_images(path):
    '''
    Loads the images in the provided path with jpg format
    :param path: directory path for images
    :return: an array of images
    '''
    images = []
    for img in os.listdir(path):
        if img.endswith('.jpg') or img.endswith('.jpeg'):
            images.append((img, cv2.imread(path+img, 0)))
    return images

# image pre-processing; resize, gray-scale
imgPath = 'C:/Users/sagar/Desktop/Images/test/negative/'                       #image path for input
imgPathOut = 'C:/Users/sagar/Desktop/NewImages/test/negative/'         #image path for output
images = load_images(imgPath)                                               #array of images

if not os.path.exists(imgPathOut):
    os.makedirs(imgPathOut)
img_num = 1

for name, i in images:
    resized_image = cv2.resize(i, (30, 30))
    cv2.imwrite(imgPathOut+str(img_num)+'.jpg', resized_image)
    img_num += 1
print(img_num)
