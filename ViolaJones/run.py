import cv2
import faceDet.ViolaJones.ViolaJones as VJ
import faceDet.ViolaJones.Integral as intImg
import faceDet.ViolaJones.Cascade as cascade
import os

"""
    This runs the whole training of the algorithm and gives a trained file with .pkl extension.
"""



def load_images(path):
    images = []
    for img in os.listdir(path):
        if img.endswith('.jpg'):
            images.append(cv2.imread(path+img, 0))
    return images

def trainBoost(t, pos, neg):
    clf = VJ.ViolaJones(T=t)
    clf.train(pos, neg)
    evaluate(clf, pos+neg)
    clf.save(str(t))

def testBoost(filename, testImg):
    clf = VJ.ViolaJones.load(filename)
    evaluate(clf, testImg)

def trainCascade(t, pos, neg):
    c = cascade.CascadeClassifier(t)
    c.train(pos, neg)
    evaluate(c, pos+neg)
    c.save("cTrain")

def testCascade(filename, testImg):
    c = cascade.CascadeClassifier.load(filename)
    evaluate(c, testImg)

def evaluate(clf, data):
    p = 0
    n = 0
    correct = 0
    for x, y in data:
        if clf.classify(x) == y:
            correct += 1
            if y == 1: p += 1
            else: n += 1

    print("Classified %d pos %d neg correctly" % (p, n))
    print("Classified %d out of %d examples: %d" % (correct, len(data), (correct*100)/len(data)))

# file paths: csv, positive, negative
print("Loading files")

# training images
posPath = 'H:/CSC485/faceDet/Images/train/positive/'
negPath = 'H:/CSC485/faceDet/Images/train/negative/'
posImg = load_images(posPath)      # has and returns array of (read(image))
negImg = load_images(negPath)      # (read(image))
posIntImages = intImg.powerfulIntegral(posImg, 1)  # array of (ii, val)
negIntImages = intImg.powerfulIntegral(negImg, 0)

trainCascade([5, 10, 20, 25], posIntImages, negIntImages)

# testing images
posPath = 'H:/CSC485/faceDet/Images/test/positive/'
negPath = 'H:/CSC485/faceDet/Images/test/negative/'
posImg = load_images(posPath)      # has and returns array of (read(image))
negImg = load_images(negPath)      # (image)
posIntImages = intImg.powerfulIntegral(posImg, 1)  # array of (ii, val)
negIntImages = intImg.powerfulIntegral(negImg, 0)

testCascade("cTrain", posIntImages+negIntImages)
