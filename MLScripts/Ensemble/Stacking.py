
"""
Stacked Generalization (Stacking)

Stacked generalization (or stacking) (Wolpert, 1992) is a different way of combining multiple models,
that introduces the concept of a meta learner. Although an attractive idea, it is less widely used than bagging and boosting.
Unlike bagging and boosting, stacking may be (and normally is) used to combine models of different types. The procedure is as follows:

1) Split the training set into two disjoint sets.
2) Train several base learners on the first part.
3) Test the base learners on the second part.
4) Using the predictions from 3) as the inputs, and the correct responses as the outputs, train a higher level learner.

Note that steps 1) to 3) are the same as cross-validation, but instead of using a winner-takes-all approach,
we combine the base learners, possibly nonlinearly.
"""
import numpy as np
from sklearn.cross_validation import StratifiedKFold

class StackedClassifier:

    def __init__(self, baseclfs, blendclf, verbose=False):
        self.baseclfs = baseclfs
        self.blendclf = blendclf
        self.verbose = verbose

    def fit(self, X, Y, useProbasInstead=False):
        self.useProbasInstead = useProbasInstead
        blendTrain = np.zeros((X.shape[0], len(self.baseclfs)))

        for i, clf in enumerate(self.baseclfs):
            if self.verbose: print("StackedClassifier : Begun training base classifier %d" % ((i+1)))
            clf.fit(X, Y)
            blendTrain = self._computeBlend(blendTrain, i, clf, X, useProbasInstead)
            if self.verbose: print("StackedClassifier : Finished training base classifier %d" % ((i+1)))

        self.blendclf.fit(blendTrain, Y)
        if self.verbose: print("StackedClassifier : Finished fitting")

    def predict(self, X, autoScale=False):
        blendPredict = np.zeros((X.shape[0], len(self.baseclfs)))

        for i, clf in enumerate(self.baseclfs):
            if self.verbose: print("StackedClassifier : Begun predicting base classifier %d" % ((i+1)))
            blendPredict = self._computeBlend(blendPredict, i, clf, X, self.useProbasInstead)
            if self.verbose: print("StackedClassifier : Finished predicting base classifier %d" % ((i+1)))

        yPred = self._predict(blendPredict)

        if autoScale:
            yPred = (yPred - yPred.min()) / (yPred.max() - yPred.min())

        if self.verbose: print("StackedClassifier : Finished predicting")
        return yPred

    def _computeBlend(self, blender, i, clf, x, useProbasInstead):
        if useProbasInstead:
            try:
                blender[:, i] = clf.predict_proba(x)[:, 1]
            except:
                print("StackedClassifier : Clf %d does not possess function 'predict_proba()'. Switching to 'predict()'" % ((i+1)))
                blender[:, i] = clf.predict(x)
        else:
            blender[:, i] = clf.predict(x)

        return blender

    def _predict(self, blender):
        if self.useProbasInstead:
            try:
                return self.blendclf.predict_proba(blender)[:, 1]
            except:
                print("StackedClassifier : StackedClassifier does not possess function 'predict_proba()'. Switching to 'predict()'")
                return self.blendclf.predict(blender)
        else:
            return self.blendclf.predict(blender)

    def _checkIfProbaExists(self, clf, X):
        try:
            res = clf.predict_proba(X)[:, 1]
            return True
        except:
            return False

class StackedClassifierCV(StackedClassifier):

    def __init__(self, baseclfs, blendclf, cvFolds=3, split=0.8, verbose=False):
        super().__init__(self, baseclfs, blendclf)
        self.cvFolds = cvFolds
        self.split = split

    def fit(self, X, Y, useprobas=True, randomState=0, verbose=False):
        np.random.seed(randomState)

        self.skf = list(StratifiedKFold(Y, self.cvFolds))
        blendTrain = np.zeros((X.shape[0], len(self.baseclfs)))

        for j, clf in enumerate(self.baseclfs):
            if self.verbose: print("StackedClassifier : Begun fitting base classifier %d" % ((j+1)))

            for i, (train, test) in enumerate(self.skf):
                if self.verbose: print("CLF %d : Fold %d" % ((j+1),(i+1)))
                X_train = X[train]
                y_train = Y[train]
                X_test = X[test]
                y_test = Y[test]
                clf.fit(X_train, y_train)

                if self._checkIfProbaExists(clf, X_test):
                    y_submission = clf.predict_proba(X_test)[:,1]
                else:
                    y_submission = clf.predict(X_test)

                blendTrain[test, j] = y_submission

                if self.verbose: print("CLF %d : Fold %d finished" % ((j+1),(i+1)))

        self.blendclf.fit(blendTrain, Y)

    def predict(self, X, autoScale=False):
        blendTest = np.zeros((X.shape[0], len(self.baseclfs)))

        for j, clf in enumerate(self.baseclfs):
            if self.verbose: print("StackedClassifier : Begun predicting base classifier %d" % ((j+1)))
            dataset_blend_test_j = np.zeros((X.shape[0], len(self.skf)))

            for i, (train, test) in enumerate(self.skf):
                if self.verbose: print("CLF %d : Fold %d" % ((j+1),(i+1)))

                if self._checkIfProbaExists(clf, X):
                    dataset_blend_test_j[:, i] = clf.predict_proba(X)[:,1]
                else:
                    dataset_blend_test_j[:, i] = clf.predict(X)

                if self.verbose: print("CLF %d : Fold %d finished" % ((j+1),(i+1)))

            blendTest[:,j] = dataset_blend_test_j.mean(1)

        yPred = self._predict(blendTest)

        if autoScale:
            yPred = (yPred - yPred.min()) / (yPred.max() - yPred.min())

        if self.verbose: print("StackedClassifier : Finished predicting")
        return yPred