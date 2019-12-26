import cv2
import faceDet.ViolaJones.Integral as ii
import faceDet.ViolaJones.Cascade as cascade
import faceDet.extras as fload

clfs = cascade.CascadeClassifier.load("cTrain")
path = "C:/Users/sagar/Desktop/CSC485/faceDet/Example/"
img = fload.load_images(path)
intImages = []

for name, i in img:                 # returns name.jpg, integral image array
    i = ii.integral_image(i)
    intImages.append((name, i))

for n, z in intImages:
    newImg = cv2.imread(path + n)
    w, h, nthg = newImg.shape

    for x in range(0, w - 30, 20):
        for y in range(0, h - 30, 20):
            subFrame = z[x:x + 30, y:y + 30]  # selecting
            if clfs.classify(subFrame) == 1:
                cv2.rectangle(newImg, (x, y), (30, 30), (255, 255, 0), 2)

    cv2.imshow("new", newImg)
    cv2.waitKey()
    cv2.destroyAllWindows()