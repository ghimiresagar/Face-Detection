import faceDet.ViolaJones.ViolaJones as VJ
import pickle

class CascadeClassifier():
    def __init__(self, layers):
        self.layers = layers
        self.clfs = []

    def train(self, pos, neg):
        l = []
        fp = []
        for feature_num in self.layers:     # each run of adaboost gives a set of weak clfs treated as strong classifier
            if len(neg) == 0:
                print("Stopping early. FPR = 0")
                break
            clf = VJ.ViolaJones(T=feature_num)
            clf.train(pos, neg)
            self.clfs.append(clf)
            false_positives = []
            for ex in neg:
                if self.classify(ex[0]) == 1:
                    false_positives.append(ex)
            neg = false_positives
            l.append(feature_num)
            fp.append(len(neg))
            print("Obtained cascade level with %d features, %d FP" % (feature_num, len(neg)))
        self.showit(l, fp)

    def showit(self, l, fp):
        for x, y in zip(l, fp):
            print("%d     %d" % (x, y))


    def classify(self, image):
        for clf in self.clfs:       # strong classifier in VJ classify (as a whole)
            if clf.classify(image) == 0:
                return 0
        return 1

    def save(self, filename):
        with open(filename + ".pkl", 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename + ".pkl", 'rb') as f:
            return pickle.load(f)