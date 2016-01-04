import pandas as pd
import numpy as np

"""
General Tips :

-- In order to locate rows in a data frame manually, according to some criteria, use:

        df.loc[ (df.col1.isnull()) | (df.col2.notnull()) & (df.col3 == someval) , "outputCol"] = somevalue
                {---------- selection criteria -------------------------------} {output feature} {-value-}

                { | : or, & : and }

-- In order to work with datetime data:

    from datetime import datetime

    dt = pd.DatetimeIndex(df["datetime"])
    df.set_index(dt, inplace=True)

    df['month'] = dt.month
    df['year'] = dt.year
    df['dow'] = dt.dayofweek
    df['woy'] = dt.weekofyear

    # Also contains dt.hour and so on. Check docs
"""

def loadData(filePath, header=0, describe=False):
    """
    Helper method to load csv files

    :param filePath: Path to csv data file
    :param header (Optional, Default = 0) : Sets header at row 0, to disable put header=None
    :param describe (Optional, Default = False) : Set to True if you want describeDataFrame() to describe the loaded data
    :return: pandas dataframe
    """
    df = pd.read_csv(filePath, header=header)
    if describe: describeDataframe(df)
    return df

def getColNames(df):
    """
    Helper method to access the column names of dataframe

    :param df: pandas dataframe
    :return: list of column names
    """
    return df.columns.tolist()

def reorderOutputRow(df, outputRowIndex=-1):
    """
    Reorders the columns of the Datafrane in order to put the output labels at the startint index

    :param df: pandas dataframe
    :param outputRowIndex (Optional, Default = -1) : Negative index of output row eg to shift from last col to first col, outputRowIndex = -1
    :return: reordered pandas dataframe
    """
    cols = getColNames(df)
    cols = cols[outputRowIndex:] + cols[:outputRowIndex]
    df = df[cols]
    return df

def describeDataframe(df):
    """
    Describes the given dataframe ie. Info, Description and DType of the given dataframe

    :param df: pandas dataframe
    """
    print(df.info(), "\n")
    print(df.describe(), "\n")
    print(df.dtypes, "\n")

def getMissingRowsOfCol(df, colName, attributes=[]):
    """
    Get all rows where a specified column is missing or null.

    :param df: pandas dataframe
    :param colName: checking column if null
    :param attributes (Optional, Default = []) : list of attributes from that row where column is null
    :return: row of all attributes, or those attributes which were specified
    """
    if len(attributes) > 0:
        return df[df[colName].isnull()][attributes]
    else:
        return df[df[colName].isnull()]

def applyFunction(df, selectedCols, outputCol, func):
    """
    Applys a function by using the selected columns as decision variables, and produces an output column that is stored
    in outputCol.

    :param df: pandas dataframe
    :param selectedCols: list of column names
    :param outputCol: string output column name
    :param func: Applying function with signature [ func(dataframe) ]. Can access selected cols variables as
                 dataframe["selectedCol"]. Must return a value of same data type as desired column data type

                 Eg: func(df):
                        data1 = df["col1"]
                        data2 = df["col2"]
                        return process(data1, data2)

    :return: pandas dataframe
    """
    df[outputCol] = df[selectedCols].apply(func, axis=1)
    return df

def mapFunction(df, selectedCol, outputCol, dictionary):
    """
    Map the values of selectedCol rows to specific values using a dict() or defaultdict() value, and produces an output
    column that is stored in outputCol

    :param df: pandas dataframe
    :param selectedCol: string column name
    :param outputCol: string output column name. Note outputCol != selectedCol
    :param dictionary: dictionary to map values from selectedCol to outputCol
    :return: pandas dataframe
    """
    df[outputCol] = df[selectedCol].map(dictionary)
    return df

def copyFeatures(df, oldfeatureName, newfeatureName):
    """
    Copies the features from old name to new name

    :param df: pandas datafrane
    :param oldfeatureName: string name of old column
    :param newfeatureName: string name of new column
    :return: pandas dataframe
    """
    df[newfeatureName] = df[oldfeatureName]
    return df

def convertToType(df, col, type):
    """
    Converts a column to some other type

    :param df: pandas dataframe
    :param col: column name
    :param type: New type. Possible types : int, float, "category", object
    :return: pandas dataframe
    """
    df[col] = df[col].astype(type)
    return df

def dropUnimportantFeatures(df, cols, dropna=False):
    """
    Drops unimportant features from dataframe

    :param df: pandas dataframe
    :param cols: list of column names to drop
    :param dropna (Optional, Default = False) : If set to true, will drop all rows that have any null values
    :return: pandas dataframe
    """
    df = df.drop(cols, axis=1)
    if dropna: df = df.dropna()
    return df

def convertPandasDataFrameToNumpyArray(df):
    """
    Converts pandas dataframe to numpy data array

    :param df: pandas dataframe
    :return: numpy array
    """
    return df.values

