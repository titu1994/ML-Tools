import numpy as np
import joblib
import glob
from copy import copy

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin


def check_module_exists(modulename):
    try:
        __import__(modulename)
    except ImportError:
        return False
    return True


if check_module_exists('keras'):
    import keras.models as keras_models
    from keras.utils.np_utils import to_categorical
    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

def get_predictions(model, X):
    if hasattr(model, 'predict_proba'): # Normal SKLearn classifiers
        pred = model.predict_proba(X)
    elif hasattr(model, '_predict_proba_lr'): # SVMs
        pred = model._predict_proba_lr(X)
    else:
        pred = model.predict(X)

    if len(pred.shape) == 1:  # for 1-d ouputs
            pred = pred[:, None]

    return pred


class ExternalStackedGeneralizer(BaseEstimator, ClassifierMixin):
    """
    Base class for stacked generalization classifier models on predictions
    which were saved into numpy arrays externally.
    """

    def __init__(self, predictions_dir, blending_models=None, n_folds=10, verbose=True):
        """
        Stacked Generalizer Classifier which trains on predictions which are
        stored in external numpy arrays.

        Trains a series of base models using K-fold cross-validation, then combines
        the predictions of each model into a set of features that are used to train
        a high-level classifier model.

        Example:
        Save predictions of various models using model.predict_proba() and obtain the numpy arrays :
        predictions = model.predict_proba(X)

        Then save the predictions in some directory with appropriate name :
        np.save(prediction_dir + 'model_name.npy', predictions)

        Once you have stored the predictions of various models, use this model
        >>> model1 = "... some model"
        >>> model2 = "... another model"
        >>> models = [model1, model2]
        >>> stack = ExternalStackedGeneralizer(predictions_dir, models, n_folds=10)
        >>> stack.fit(None, y) # Note, X isnt required here
        >>> stack.predict(X)

        Parameters
        -----------
        predictions_dir: path of the folder with the saved numpy predictions
        blending_model: object
            A classifier model used to aggregate the outputs of the trained base
            models. Must have a .fit and .predict_proba/.predict method
        n_folds: int
            The number of K-folds to use in =cross-validated model training
        verbose: boolean
        """
        self.predictions_dir = predictions_dir
        self.blending_models = blending_models
        self.n_folds = n_folds
        self.verbose = verbose

    def fit(self, X, y):
        X_blend = self._fitTransformBaseModels()
        self._fitBlendingModel(X_blend, y)

    def predict(self, X):
        # perform model averaging to get predictions
        X_blend = self._transformBaseModels()
        predictions = self._transformBlendingModel(X_blend)
        pred_classes = np.argmax(predictions, axis=1)
        return pred_classes

    def predict_proba(self, X):
        # perform model averaging to get predictions
        X_blend = self._transformBaseModels()
        predictions = self._transformBlendingModel(X_blend)
        return predictions

    def evaluate(self, y, y_pred):
        print(classification_report(y, y_pred))
        print('Confusion Matrix:')
        print(confusion_matrix(y, y_pred))
        return(accuracy_score(y, y_pred))

    def _transformBaseModels(self):
        # predict via model averaging
        predictions = []

        base_dir = self.predictions_dir
        path = base_dir + "*.npy"

        files = glob.glob(path)
        for file in files:
            if self.verbose: print('Loading numpy file %s' % (file))
            cv_predictions = np.load(file)
            predictions.append(cv_predictions.mean(axis=0))

        # concat all features
        predictions = np.hstack(predictions)
        if self.verbose: print('Loaded predictions. Shape : ', predictions.shape)
        return predictions

    def _fitTransformBaseModels(self):
        return self._transformBaseModels()

    def _fitBlendingModel(self, X_blend, y):
        for model_id, blend_model in enumerate(self.blending_models):
            if self.verbose:
                model_name = "%s" % blend_model.__repr__()
                print('Fitting Blending Model:\n%s' % model_name)

            self.blending_model_cv = []

            for j, (train_idx, test_idx) in enumerate(StratifiedKFold(self.n_folds, shuffle=True, random_state=1000)):
                if self.verbose:
                    print('Fold %d' % j)

                X_train, y_train = X_blend[train_idx], y[train_idx]
                X_test, y_test = X_blend[test_idx], y[test_idx]

                model = copy(blend_model)

                if check_module_exists('keras'):
                    if isinstance(model, keras_models.Model) or isinstance(model, keras_models.Sequential):
                        model_path = 'models/stack/keras_model_%d_cv_%d' % (model_id + 1, j + 1)
                        checkpoint = ModelCheckpoint(model_path,
                                                     monitor='val_fbeta_score', verbose=1,
                                                     save_best_only=True, save_weights_only=True,
                                                     mode='max')

                        reduce_lr = ReduceLROnPlateau(monitor='val_fbeta_score', patience=5, mode='max',
                                                      factor=0.8, cooldown=5, min_lr=1e-6, verbose=2)

                        y_train_categorical = to_categorical(y_train, 3)
                        y_test_categorical = to_categorical(y_test, 3)

                        model.fit(X_train, y_train_categorical, batch_size=128, nb_epoch=50, callbacks=[checkpoint, reduce_lr],
                                  validation_data=(X_test, y_test_categorical))

                        model.load_weights(model_path)

                        preds = model.predict(X_test, batch_size=128)
                        preds = np.argmax(preds, axis=1)

                        score = f1_score(y_test, preds)
                        print('Keras Model %d - CV %d Score : %0.3f' % (model_id + 1, j + 1, score))

                        # add trained model to list of CV'd models
                        self.blending_model_cv.append(model)
                        continue

                model_path = 'models/stack/sklearn_model_%d_cv_%d' % (model_id + 1, j + 1)
                model.fit(X_train, y_train)

                preds = get_predictions(model, X_test)
                preds = np.argmax(preds, axis=1)

                score = f1_score(y_test, preds)
                print('SKLearn Model %d - CV %d Score : %0.3f' % (model_id + 1, j + 1, score))

                joblib.dump(model, model_path)
                # add trained model to list of CV'd models
                self.blending_model_cv.append(model)


    def _transformBlendingModel(self, X_blend):
        # make predictions from averaged models
        cv_predictions = None
        n_models = len(self.blending_model_cv)

        for i, model in enumerate(self.blending_model_cv):
            cv_predictions = None
            model_predictions = get_predictions(model, X_blend)

            if cv_predictions is None:
                cv_predictions = np.zeros((n_models, X_blend.shape[0], model_predictions.shape[1]))

            cv_predictions[i,:,:] = model_predictions

        # perform model averaging to get predictions
        predictions = cv_predictions.mean(0)
        return predictions
