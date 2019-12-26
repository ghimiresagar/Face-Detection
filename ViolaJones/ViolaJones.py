import numpy as np
import math
import pickle
from sklearn.feature_selection import SelectPercentile, f_classif
import faceDet.ViolaJones.Feature as weakFeature

class ViolaJones:
    def __init__(self, T=10):
        """
          Args:
            T: The number of times to loop and train the selected weak classifiers which should be used
        """
        self.T = T
        self.alphas = []
        self.clfs = []

    def train(self, posImg, negImg):
        """
            Trains the Viola Jones classifier using boosting.

        :param posImg: List of Positive Images, (ndarray of images)
        :param negImg: List of Negative Images, (ndarray of images)
        :param pos_num: number of positive samples
        :param neg_num: number of negative samples
        :return:
        """

        posNum = len(posImg)
        negNum = len(negImg)

        training_data = posImg+negImg
        weights = np.zeros(len(training_data))
        print("Computing weights of images")
        for x in range(len(training_data)):
            if training_data[x][1] == 1:
                weights[x] = 1.0 / (2 * posNum)
            else:
                weights[x] = 1.0 / (2 * negNum)

        print("Building features")
        features = weakFeature.computeFeatures(training_data[0][0].shape)
        print("Applying features to training examples")
        X, y = weakFeature.apply_features(features, training_data)
        print("Selecting best features")

        # bases on number of highest scores for each features, select the best % of features
        indices = SelectPercentile(f_classif, percentile=10).fit(X.T, y).get_support(indices=True) # returns an array
        X = X[indices]  # selecting best features row through returned indices
        features = features[indices]    # new features is best features only
        print("Selected %d potential features" % len(X))

        for t in range(self.T):
            """
            Normalize weights
            Train weak classifiers selected
            get errors, beta value
            get alpha value
            select the classifiers as strong classifiers
            """
            weights = weights / np.linalg.norm(weights)  # normalize weights
            weak_classifiers = self.train_weak(X, y, features, weights)  # train weak classifier
            clf, error, accuracy = self.select_best(weak_classifiers, weights, training_data)
            beta = error / (1.0 - error)
            for i in range(len(accuracy)):
                weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
            alpha = math.log(1.0 / beta)
            self.alphas.append(alpha)
            self.clfs.append(clf)
            print("Chose classifier: %s with accuracy: %f and alpha: %f" % (
                str(clf), len(accuracy) - sum(accuracy), alpha))


    def train_weak(self, X, y, features, weights):
        """
            this algorithm is designed to select a single rectangle feature which best separates the positive and negative
        examples with respect to weights of images; using min error to calculate error by feature rather than weighted error
        to compute error at constant time

        :param X: Selected features for training
        :param y: actual positive or negative representation of images
        :param features: selected best features to get the feature at the end
        :param weights: weights associated with training data
        :return: array of weak classifiers
        """
        totalPostW, totalNegW = 0, 0     # total weights
        for w, label in zip(weights, y):        # this has to happen for every training to get weights
            if label == 1:
                totalPostW += w          # sum of positive weight
            else:
                totalNegW += w          # sum of neg weight

        classifiers = []        # stores my classifiers
        total_features = X.shape[0]     # no. of features

        # this loop goes through all the features and selects couple of best p, t with minimum error
        for index, feature in enumerate(X):     # X is 2d array, so we give those values an index to go through it
            # if len(classifiers) % 200 == 0 and len(classifiers) != 0:     # print every nth classifier
            #     print("Trained %d classifiers out of %d" % (len(classifiers), total_features))

            # returns list of tuples
            applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])  # sorting with respect to feature

            pos_seen, neg_seen = 0, 0   # num of pos or neg seen
            pos_weights, neg_weights = 0, 0     # pos or neg weights seen
            min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None

            # this loop finds the min error, threshold and polarity of the feature to be selected if has min error
            for w, f, label in applied_feature:     # for this feature 'fi' with weight 'wi' and label 'li'
                error = min(neg_weights + totalPostW - pos_weights, pos_weights + totalNegW - neg_weights)
                if error < min_error:   # at first it's always less, next time it might be or not
                    min_error = error
                    best_feature = features[index]  # feature at the index with pos and neg region
                    best_threshold = f  # feature value at this min error gives best threshold
                    best_polarity = 1 if pos_seen > neg_seen else -1    # if more positive seen, it's positively polar

                if label == 1:      # pos image
                    pos_seen += 1   # num of positive images seen and weight increases
                    pos_weights += w
                else:
                    neg_seen += 1   # neg image, num of neg images seen and neg weight increases
                    neg_weights += w

            clf = weakFeature.WeakClassifier(best_feature[0], best_feature[1], best_threshold, best_polarity)
            classifiers.append(clf)
        return classifiers

    def select_best(self, classifiers, weights, training_data):
        """
        Selects the best weak classifier for the given weights
          Args:
            classifiers: An array of weak classifiers
            weights: An array of weights corresponding to each training example
            training_data: An array of tuples. The first element is the numpy array of shape (m, n) representing the integral image. The second element is its classification (1 or 0)
          Returns:
            A tuple containing the best classifier, its error, and an array of its accuracy on images
        """
        best_clf, best_error, best_accuracy = None, float('inf'), None
        for clf in classifiers:
            error, accuracy = 0, []
            for data, w in zip(training_data, weights):
                correctness = abs(clf.classify(data[0]) - data[1])     # if correct should be 0
                accuracy.append(correctness)
                error += w * correctness
            error = error / len(training_data)
            if error < best_error:
                best_clf, best_error, best_accuracy = clf, error, accuracy
        return best_clf, best_error, best_accuracy

    def classify(self, image):
        total = 0
        for alpha, clf in zip(self.alphas, self.clfs):
            total += alpha * clf.classify(image)
        return 1 if total >= 0.5 * sum(self.alphas) else 0

    def save(self, filename):
        """
        Saves the classifier to a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename + ".pkl", 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        A static method which loads the classifier from a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename + ".pkl", 'rb') as f:
            return pickle.load(f)