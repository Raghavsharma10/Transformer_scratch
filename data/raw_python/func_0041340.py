def Create_Random_Forest(train):
    """
    Fits Random Forest to training set.
    
    :param train: This is the file name of a csv file we wish to have fitted to a Random Forest, does not need to have features already extracted.
    :returns: Returns sklearn.ensemble.Random_Forest_Classifier fitted to training set.
    """
    trainDF=pd.read_csv(train)
    train=Feature_Engineering(train,trainDF)
    RF = RFC(min_samples_split=10, n_estimators= 700, criterion= 'gini', max_depth=None)
    RF.fit(train.iloc[:, 1:], train.iloc[:, 0])
    return RF