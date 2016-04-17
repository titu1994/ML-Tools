import numpy as np
from sklearn.cross_validation import KFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from copy import copy

def getPredictions(model, X):
    if hasattr(model, 'predict_proba'):
        pred = model.predict_proba(X)
    else:
        pred = model.predict(X)

    if len(pred.shape) == 1:  # for 1-d ouputs
            pred = pred[:,None]

    return pred


def checkModuleExists(modulename):
    try:
        __import__(modulename)
    except ImportError:
        return False
    return True


class StackedGeneralizer(BaseEstimator, ClassifierMixin):
    """Base class for stacked generalization classifier models
    """

    def __init__(self, base_models=None, blending_model=None, n_folds=5, verbose=True):
        """
        Stacked Generalizer Classifier

        Trains a series of base models using K-fold cross-validation, then combines
        the predictions of each model into a set of features that are used to train
        a high-level classifier model.
        Parameters
        -----------
        base_models: list of classifier models
            Each model must have a .fit and .predict_proba/.predict method a'la
            sklearn
        blending_model: object
            A classifier model used to aggregate the outputs of the trained base
            models. Must have a .fit and .predict_proba/.predict method
        n_folds: int
            The number of K-folds to use in =cross-validated model training
        verbose: boolean
        """
        self.base_models = base_models
        self.blending_model = blending_model
        self.n_folds = n_folds
        self.verbose = verbose
        self.base_models_cv = None

    def fit(self, X, y):
        X_blend = self._fitTransformBaseModels(X, y)
        self._fitBlendingModel(X_blend, y)

    def fit_nn(self, X, y, batch_size=128, nb_epochs=100, show_accuracy=True, verbose=0, callback=[], validation_split=0.,
               validation_data=None):
        if not checkModuleExists("keras"):
            self.fit(X, y)
        else:
            X_blend = self._fitTransformBaseModels(X, y)
            self._fitBlendingModel_nn(X_blend, y, batch_size, nb_epochs, show_accuracy, verbose, callback, validation_split,
               validation_data)

    def predict(self, X):
        # perform model averaging to get predictions
        X_blend = self._transformBaseModels(X)
        predictions = self._transformBlendingModel(X_blend)
        pred_classes = [np.argmax(p) for p in predictions]
        return pred_classes

    def predict_proba(self, X):
        # perform model averaging to get predictions
        X_blend = self._transformBaseModels(X)
        predictions = self._transformBlendingModel(X_blend)

        return predictions

    def evaluate(self, y, y_pred):
        print(classification_report(y, y_pred))
        print('Confusion Matrix:')
        print(confusion_matrix(y, y_pred))
        return(accuracy_score(y, y_pred))

    def _fitBaseModels(self, X, y):
        if self.verbose:
            print('Fitting Base Models...')

        kf = list(KFold(y.shape[0], self.n_folds))

        self.base_models_cv = {}

        for i, model in enumerate(self.base_models):

            model_name = "model %02d: %s" % (i+1, model.__repr__())
            if self.verbose:
                print('Fitting %s' % model_name)

            # run stratified CV for each model
            self.base_models_cv[model_name] = []
            for j, (train_idx, test_idx) in enumerate(kf):
                if self.verbose:
                    print('Fold %d' % (j + 1))

                X_train = X[train_idx]
                y_train = y[train_idx]

                model.fit(X_train, y_train)

                # add trained model to list of CV'd models
                self.base_models_cv[model_name].append(copy(model))

    def _transformBaseModels(self, X):
        # predict via model averaging
        predictions = []
        for key in sorted(self.base_models_cv.keys()):
            cv_predictions = None
            n_models = len(self.base_models_cv[key])
            for i, model in enumerate(self.base_models_cv[key]):
                model_predictions = getPredictions(model, X)

                if cv_predictions is None:
                    cv_predictions = np.zeros((n_models, X.shape[0], model_predictions.shape[1]))

                cv_predictions[i,:,:] = model_predictions

            # perform model averaging and add to features
            predictions.append(cv_predictions.mean(0))

        # concat all features
        predictions = np.hstack(predictions)
        return predictions

    def _fitTransformBaseModels(self, X, y):
        self._fitBaseModels(X, y)
        return self._transformBaseModels(X)

    def _fitBlendingModel(self, X_blend, y):
        if self.verbose:
            model_name = "%s" % self.blending_model.__repr__()
            print('Fitting Blending Model:\n%s' % model_name)

        kf = list(KFold(y.shape[0], self.n_folds))
        # run  CV
        self.blending_model_cv = []

        for j, (train_idx, test_idx) in enumerate(kf):
            if self.verbose:
                print('Fold %d' % j)

            X_train = X_blend[train_idx]
            y_train = y[train_idx]

            model = copy(self.blending_model)

            model.fit(X_train, y_train)

            # add trained model to list of CV'd models
            self.blending_model_cv.append(model)

    def _fitBlendingModel_nn(self, X_blend, y, batch_size=128, nb_epochs=100, show_accuracy=True, verbose=0, callback=[], validation_split=0.,
               validation_data=None):
        if self.verbose:
            model_name = "%s" % self.blending_model.__repr__()
            print('Fitting Blending Model:\n%s' % model_name)

        kf = list(KFold(y.shape[0], self.n_folds))
        # run  CV
        self.blending_model_cv = []

        for j, (train_idx, test_idx) in enumerate(kf):
            if self.verbose:
                print('Fold %d' % j)

            X_train = X_blend[train_idx]
            y_train = y[train_idx]

            model = copy(self.blending_model)

            model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epochs, show_accuracy=show_accuracy,
                      verbose=verbose, callback=callback, validation_split=validation_split, validation_data=validation_data)

            # add trained model to list of CV'd models
            self.blending_model_cv.append(model)

    def _transformBlendingModel(self, X_blend):
        global cv_predictions

        # make predictions from averaged models
        predictions = []
        n_models = len(self.blending_model_cv)
        for i, model in enumerate(self.blending_model_cv):
            cv_predictions = None
            model_predictions = getPredictions(model, X_blend)

            if cv_predictions is None:
                cv_predictions = np.zeros((n_models, X_blend.shape[0], model_predictions.shape[1]))

            cv_predictions[i,:,:] = model_predictions

        # perform model averaging to get predictions
        predictions = cv_predictions.mean(0)
        return predictions


if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression

    VERBOSE = True
    N_FOLDS = 5

    # load data and shuffle observations
    data = load_digits()
    X = data.data
    y = data.target
    shuffle_idx = np.random.permutation(y.size)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    # hold out 20 percent of data for testing accuracy
    n_train = round(X.shape[0]*.8)
    # define base models
    base_models = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
                       RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
                       ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),]
    # define blending model
    blending_model = LogisticRegression()
    # initialize multi-stage model
    sg = StackedGeneralizer(base_models, blending_model,
                                n_folds=N_FOLDS, verbose=VERBOSE)

    # fit model
    sg.fit(X[:n_train],y[:n_train])

    # test accuracy
    pred_probas = sg.predict_proba(X[n_train:])
    preds = sg.predict(X[n_train:])
    print("PostProcessed Predictions Class : ", preds)
    _ = sg.evaluate(y[n_train:], preds)

    sg.verbose = False