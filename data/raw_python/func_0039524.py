def return_features_numpy_base(dbpath, set_object, points_amt, names):
    """
    Generic function which returns a 2d numpy array of extracted features

    Parameters
    ----------
    dbpath : string, path to SQLite database file
    set_object : object (either TestSet or TrainSet) which is stored in the database
    points_amt : int, number of data points in the database
    names : list of strings, a list of feature names which are to be retrieved from the database, if equal to 'all',
    all features will be returned

    Returns
    -------
    return_array : ndarray of features, each row corresponds to a single datapoint. If a single feature
    is a 1d numpy array, then it will be unrolled into the resulting array. Higher-dimensional numpy arrays are not
    supported.
    """
    engine = create_engine('sqlite:////' + dbpath)
    session_cl = sessionmaker(bind=engine)
    session = session_cl()
    tmp_object = session.query(set_object).get(1)

    if names == 'all':
        columns_amt = 0
        for feature in tmp_object.features:
            if type(tmp_object.features[feature]) is np.ndarray:
                columns_amt += tmp_object.features[feature].shape[0]
            else:
                columns_amt += 1
        return_array = np.zeros([points_amt, columns_amt])
        for i in enumerate(session.query(set_object).order_by(set_object.id)):
            counter = 0
            for feature in i[1].features:
                feature_val = i[1].features[feature]
                if type(feature_val) is np.ndarray:
                    columns_amt = feature_val.shape[0]
                    return_array[i[0], counter:counter + columns_amt] = feature_val[:]
                    counter += feature_val.shape[0]
                else:
                    return_array[i[0], counter] = feature_val
                    counter += 1
    else:
        columns_amt = 0
        for feature in tmp_object.features:
            if feature in names:
                if type(tmp_object.features[feature]) is np.ndarray:
                    columns_amt += tmp_object.features[feature].shape[0]
                else:
                    columns_amt += 1
        return_array = np.zeros([points_amt, columns_amt])
        for i in enumerate(session.query(set_object).order_by(set_object.id)):
            counter = 0
            for feature in i[1].features:
                if feature in names:
                    feature_val = i[1].features[feature]
                    if type(feature_val) is np.ndarray:
                        columns_amt = feature_val.shape[0]
                        return_array[i[0], counter:counter + columns_amt] = feature_val[:]
                        counter += feature_val.shape[0]
                    else:
                        return_array[i[0], counter] = feature_val
                        counter += 1
    session.close()
    return return_array