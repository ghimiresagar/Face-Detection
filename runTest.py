import cv2
import faceDet.ViolaJones.Integral as ii
import faceDet.ViolaJones.Cascade as cascade
import faceDet.extras as fload

clfs = cascade.CascadeClassifier.load("cTrain")
path = "C:/Users/sagar/Desktop/CSC485/faceDet/Example/neg/"
img = fload.load_images(path)
intImages = []

for name, i in img:                 # returns name.jpg, integral image array
    i = ii.integral_image(cv2.resize(i, (30, 30)))
    intImages.append((name, i))

for n, z in intImages:
    newImg = cv2.imread(path + n)
    w, h, nthg = newImg.shape
    if clfs.classify(z) == 1:
        cv2.rectangle(newImg, (0, 0), (h, w), (255, 255, 0), 2)
    else:
        cv2.rectangle(newImg, (0, 0), (h, w), (0, 0, 255), 2)
    cv2.imshow("new", newImg)
    cv2.waitKey()
    cv2.destroyAllWindows()