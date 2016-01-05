
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


def writeOutputFile(filename, headerRow, zippedRows):
    """
    Writes the output of the rows into the filename specified

    :param filename: Output filename
    :param headingRow: Header row
    :param zippedRows: zipped file of all the output rows
    :return:
    """
    import csv

    f = open(filename + ".csv", "w", newline="")
    csvWriter = csv.writer(f)
    csvWriter.writerow(headerRow)

    csvWriter.writerows(zip(*zippedRows))
    f.close()

def checkModuleExists(modulename):
    try:
        __import__(modulename)
    except ImportError:
        return False
    return True
