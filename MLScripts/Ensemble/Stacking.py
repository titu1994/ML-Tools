
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
from MLScripts.Helpers import checkModuleExists
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
import sklearn.linear_model as linear

def createBlendingClassifier(solver="liblinear", tol=1e-10, fit_intercepts=True, cv=None,):
    if cv != None:
        return linear.LogisticRegression(tol=tol, fit_intercept=fit_intercepts, solver=solver, )
    else:
        return linear.LogisticRegressionCV(cv=cv, tol=tol, fit_intercept=fit_intercepts,)

def createBlendingRegressor(solver="liblinear", tol=1e-10, cv=None,):
    if cv != None:
        return createBlendingClassifier(solver, tol, fit_intercepts=False, )
    else:
        return createBlendingClassifier(solver, tol, fit_intercepts=False, cv=cv)

class StackedClassifier:

    def __init__(self, baseclfs, blendclf, verbose=False):

        clfs = []
        for clf in baseclfs:
            if hasattr(clf, 'predict_proba'):
                clfs.append(clf)
            else:
                calibratedCLF = CalibratedClassifierCV(clf)
                clfs.append(calibratedCLF)

        self.baseclfs = clfs
        self.blendclf = blendclf

        if not hasattr(blendclf, 'predict_proba'):
            self.blendclf_has_predict_proba = False
        else: self.blendclf_has_predict_proba = True

        self.verbose = verbose

        if checkModuleExists("xgboost"):
            import xgboost as xgb
            self.xgbclassifier = xgb.XGBClassifier
            self.importedXGB = True
        else:
            self.importedXGB = False

    def fit(self, X, Y, xgb_eval_metric=None, xgb_eval_set=None, xgb_early_stopping_rounds=None):
        blendTrain = np.zeros((X.shape[0], len(self.baseclfs)))

        for i, clf in enumerate(self.baseclfs):
            if self.verbose: print("StackedClassifier : Begun training base classifier %d" % ((i+1)))

            if self.importedXGB:
                if isinstance(clf, self.xgbclassifier):
                    clf.fit(X, Y, verbose=self.verbose, eval_metric=xgb_eval_metric, eval_set=xgb_eval_set,
                            early_stopping_rounds=xgb_early_stopping_rounds)
                else:
                    clf.fit(X, Y)
            else:
                clf.fit(X, Y)

            if self.verbose: print("StackedClassifier : Begun training the Blending Classifier")
            blendTrain = self._computeBlend(blendTrain, i, clf, X)
            if self.verbose: print("StackedClassifier : Finished training Blend Classifier")

            if self.verbose: print("StackedClassifier : Finished training base classifier %d" % ((i+1)))

        self.blendclf.fit(blendTrain, Y)
        if self.verbose: print("StackedClassifier : Finished fitting")

    def predict(self, X, autoScale=False):
        blendPredict = np.zeros((X.shape[0], len(self.baseclfs)))

        for i, clf in enumerate(self.baseclfs):
            if self.verbose: print("StackedClassifier : Begun predicting base classifier %d" % ((i+1)))
            blendPredict = self._computeBlend(blendPredict, i, clf, X)
            if self.verbose: print("StackedClassifier : Finished predicting base classifier %d" % ((i+1)))

        if self.verbose: print("StackedClassifier : Begun predicting with the Blending Classifier")
        yPred = self._predict(blendPredict)
        if self.verbose: print("StackedClassifier : Finished predicting Blend Classifier")

        if autoScale:
            yPred = (yPred - yPred.min()) / (yPred.max() - yPred.min())

        if self.verbose: print("StackedClassifier : Finished predicting")
        return yPred

    def predict_proba(self, X, autoScale=False):
        blendPredict = np.zeros((X.shape[0], len(self.baseclfs)))

        for i, clf in enumerate(self.baseclfs):
            if self.verbose: print("StackedClassifier : Begun predicting base classifier %d" % ((i+1)))
            blendPredict = self._computeBlend(blendPredict, i, clf, X)
            if self.verbose: print("StackedClassifier : Finished predicting base classifier %d" % ((i+1)))

        if self.verbose: print("StackedClassifier : Begun predicting with the Blending Classifier")
        yPred = self._predict_proba(blendPredict)
        if self.verbose: print("StackedClassifier : Finished predicting Blend Classifier")

        if autoScale:
            yPred = (yPred - yPred.min()) / (yPred.max() - yPred.min())

        if self.verbose: print("StackedClassifier : Finished predicting")
        return yPred

    def _computeBlend(self, blender, i, clf, x):
        blender[:, i] = clf.predict_proba(x)[:, 1]
        return blender

    def _predict(self, blender):
        return self.blendclf.predict(blender)

    def _predict_proba(self, blender):
        if self.blendclf_has_predict_proba:
            return self.blendclf.predict_proba(blender)[:, 1]
        else:
            print("StackedClassifier : Blending CLF does not possess the attribute 'predict_proba'. Switching to predict()")
            return self._predict(blender)

    def score(self, X, y):
        return self.blendclf.score(X, y)


class StackedClassifierCV(StackedClassifier):

    def __init__(self, baseclfs, blendclf, cvFolds=3, split=0.8, verbose=False):
        super().__init__(baseclfs, blendclf, verbose=verbose)
        self.cvFolds = cvFolds
        self.split = split

    def fit(self, X, Y, useProbasInstead=True, xgb_eval_metric=None, xgb_eval_set=None, xgb_early_stopping_rounds=None, randomState=0):
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
                y_submission = clf.predict_proba(X_test)[:,1]
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

                dataset_blend_test_j[:, i] = clf.predict_proba(X)[:,1]

                if self.verbose: print("CLF %d : Fold %d finished" % ((j+1),(i+1)))

            blendTest[:,j] = dataset_blend_test_j.mean(1)

        yPred = self._predict(blendTest)

        if autoScale:
            yPred = (yPred - yPred.min()) / (yPred.max() - yPred.min())

        if self.verbose: print("StackedClassifier : Finished predicting")
        return

    def predict_proba(self, X, autoScale=False):
        blendTest = np.zeros((X.shape[0], len(self.baseclfs)))

        for j, clf in enumerate(self.baseclfs):
            if self.verbose: print("StackedClassifier : Begun predicting base classifier %d" % ((j+1)))
            dataset_blend_test_j = np.zeros((X.shape[0], len(self.skf)))

            for i, (train, test) in enumerate(self.skf):
                if self.verbose: print("CLF %d : Fold %d" % ((j+1),(i+1)))

                dataset_blend_test_j[:, i] = clf.predict_proba(X)[:,1]

                if self.verbose: print("CLF %d : Fold %d finished" % ((j+1),(i+1)))

            blendTest[:,j] = dataset_blend_test_j.mean(1)

        yPred = self._predict_proba(blendTest)

        if autoScale:
            yPred = (yPred - yPred.min()) / (yPred.max() - yPred.min())

        if self.verbose: print("StackedClassifier : Finished predicting")
        return yPred