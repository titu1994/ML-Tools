import numpy as np

class WeightedRegressor:

    def __init__(self, models, weights=[], verbose=False):

        # Initialise weights to all 1 if no weights provided
        if len(weights) == 0:
            for i in range(len(models)):
                weights.append(1)

        assert(len(models) == len(weights))

        self.models = models
        self.weights = weights
        self.verbose = verbose

    def fit(self, X, y):
        if self.verbose: print("WeightedRegressor : Begin Fitting")

        for i, model in enumerate(self.models):
            model.fit(X, y)
            if self.verbose: print("WeightedRegressor : Finished fitting regressor %d" % (i+1))

        if self.verbose: print("WeightedRegressor : Finished Fitting")

    def predict(self, X):
        if self.verbose: print("WeightedRegressor : Begin Predicting")

        ypredTemp = np.zeros((len(self.models), X.shape[0]))
        yPred = np.zeros((X.shape[0],))

        for i, model in enumerate(self.models):
            ypredTemp[i] = model.predict(X)
            if self.verbose: print("WeightedRegressor : Finished predicting regressor %d" % (i+1))

        for i in range(X.shape[0]):
            for j in range(len(self.models)):
                yPred[i] += self.weights[j] * ypredTemp[j][i]

            yPred[i] = yPred[i] / sum(self.weights)

        if self.verbose: print("WeightedRegressor : Finished predicting values")

        return yPred

    def rescale(self, yTrue, yPredictions):
        min_y_pred = min(yPredictions)
        max_y_pred = max(yPredictions)
        min_y_train = min(yTrue)
        max_y_train = max(yTrue)

        for i in range(len(yPredictions)):
            yPredictions[i] = min_y_train + (((yPredictions[i] - min_y_pred)/(max_y_pred - min_y_pred))*(max_y_train - min_y_train))

        if self.verbose: print("WeightedRegressor : Finished scaling predictions")
        return yPredictions


