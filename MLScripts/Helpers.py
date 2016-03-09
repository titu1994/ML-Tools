
def printDecisionTree(fn, decisionTree, featureNames=None, opClassNames=None):
    """
    Creates a pdf for the given Decision Tree from sklearn.tree package

    :param fn: filename
    :param decisionTree: decision tree object
    :param featureNames (Optional, Default = None) : list of features used by the decision tree
    :param opClassNames (Optional, Default = None) : list of output class names
    """
    from subprocess import check_call
    import sklearn.tree as tree

    with open(fn + ".dot", "w") as file:
        tree.export_graphviz(decisionTree, out_file=file,feature_names=featureNames, class_names=opClassNames, filled=True, rounded=True)

    check_call(["dot", "-Tpdf", fn + ".dot", "-o", fn + ".pdf"])

def printXGBoostTree(fn, xgBoostTree, numTrees=2, yesColor='#0000FF', noColor='#FF0000'):
    """
    Creates a pdf for the given XGBoost Tree from xgboost package

    :param fn: filename
    :param xgBoostTree: XGBoost tree object
    :param numTrees (Optional, Default = 2) : Number of decision trees to draw
    :param yesColor (Optional, Default = '#0000FF') : Color of correct output classes
    :param noColor (Optional, Default = '#FF0000'): Color of wrong output classes
    """
    from subprocess import check_call
    if checkModuleExists("xgboost"):
        import xgboost as xgb
    else:
        print("Requires xgboost library. Cannot print xgboost tree")
        return

    with open(fn + ".dot", "w") as file:
        val = xgb.to_graphviz(xgBoostTree, num_trees=numTrees, yes_color=yesColor, no_color=noColor)

    val.save(fn + ".dot")
    check_call(["dot", "-Tpdf", fn + ".dot", "-o", fn + ".pdf"])


def printFeatureImportances(featurenames, featureImportances):
    """
    Prints the feature importances of classifiers or regressors in the sklearn package

    :param featurenames: list of feature names
    :param featureImportances: list of feature importance values eg. decisionTree.feature_importances_
    """
    featureImportances = [(feature, importance) for feature, importance in zip(featurenames, featureImportances)]
    featureImportances = sorted(featureImportances, key=lambda x: x[1], reverse=True)
    print("Feature Importances : \n", featureImportances)

def printXGBFeatureImportances(featurenames, xgbTree):
    """
    Prints the feature importances of Classifiers or Regressors in the xgboost package

    :param featurenames: list of feature names
    :param xgbTree: XGBTree
    """
    snsAvailable = False
    if checkModuleExists("seaborn"):
        import seaborn as sns
        sns.set_style("white")
        snsAvailable = True
    else:
        import matplotlib.pyplot as plt

    if checkModuleExists("xgboost"):
        import xgboost as xgb
    else:
        print("Requires xgboost library. Cannot print xgboost tree importances")
        return

    featureNames = featurenames
    with open("tempfmap.fmap", "w") as f:
        for i, feature in enumerate(featureNames):
            f.write("%d\t%s\tq\n" % (i, feature))

    xgb.plot_importance(xgbTree.booster().get_fscore(fmap="tempfmap.fmap"), )
    if snsAvailable: sns.plt.show()
    else: plt.show()

    import os
    os.remove("tempfmap.fmap")


def writeOutputFile(filename, headerColumns, submissionRowsList, dtypes):
    """
    Writes the output of the rows into the filename specified

    :param filename: Output filename
    :param headingRow: list of header column names
    :param zippedRows: list of lists.
                       Ex: df["A"] and yPredicted must be written.
                           Then,
                           [df["A"], yPredicted] must be this argument
    :param dtypes: list of string dtypes that are to be cast.
                   Possible dtypes are :
                   int, int8, int16, int32, int64,
                   float, float32, float64,
                   object, -> (This is the type for string type data),
                   category -> Special type to denote importantce of int or float var,
                   bool,
                   datetime, datetime64
    """
    import pandas as pd

    submission = pd.DataFrame()
    if len(headerColumns) == len(submissionRowsList):
        for i in range(len(headerColumns)):
            submission[headerColumns[i]] = submissionRowsList[i]
            submission[headerColumns[i]] = submission[headerColumns[i]].astype(dtypes[i])

        submission.to_csv(filename, index=False)
    else:
        print("Number of headerColumns not same as number of lists of rows that must be written as o/p file")

def getClfFeatureImportances(X, y, max_features=None, n_estimators=100, random_state=0) -> list:
    """
    Utilizes ExtraTreeClassifier to determine best features

    Use :
    bestFeatureIndices = getClfFeatureIndices(X, y)
    df = df[df.columns[bestFeatureIndices]]

    :param X: Numpy array
    :param y: Numpy array
    :param max_features: maximum number of features required. If None, will return all features
    :param n_estimators: number of estimators for ExtraTreeClassifier. Increase if very large number of features
    :param random_state:
    :return: list of max_feature / all feature importances in sorted order
    """
    from sklearn.ensemble import ExtraTreesClassifier
    import numpy as np

    numFeatures = X.shape[1]
    if max_features > numFeatures: max_features = numFeatures
    if max_features == None: max_features = numFeatures

    model = ExtraTreesClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X, y)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Feature ranking:")
    bestFeatureIndices = []

    for f in range(numFeatures):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        bestFeatureIndices.append(indices[f])

    bestFeatureIndices = bestFeatureIndices[:max_features]
    return bestFeatureIndices

def getRegFeatureImportances(X, y, max_features=None, n_estimators=100, random_state=0) -> list:
    """
    Utilizes ExtraTreeRegressor to determine best features

    Use :
    bestFeatureIndices = getClfFeatureIndices(X, y)
    df = df[df.columns[bestFeatureIndices]]

    :param X: Numpy array
    :param y: Numpy array
    :param max_features: maximum number of features required. If None, will return all features
    :param n_estimators: number of estimators for ExtraTreeClassifier. Increase if very large number of features
    :param random_state:
    :return: list of max_feature / all feature importances in sorted order
    """
    from sklearn.ensemble import ExtraTreesRegressor
    import numpy as np

    numFeatures = X.shape[1]
    if max_features > numFeatures: max_features = numFeatures
    if max_features == None: max_features = numFeatures

    model = ExtraTreesRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X, y)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Feature ranking:")
    bestFeatureIndices = []

    for f in range(numFeatures):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        bestFeatureIndices.append(indices[f])

    bestFeatureIndices = bestFeatureIndices[:max_features]
    return bestFeatureIndices

def checkModuleExists(modulename):
    try:
        __import__(modulename)
    except ImportError:
        return False
    return True

