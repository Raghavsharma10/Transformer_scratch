def return_feature_list_numpy_base(dbpath, set_object):
    """
    Generic function which returns a list of tuples containing, each containing the name of the feature
    and the length of the corresponding 1d numpy array of the feature (or length of the list)

    Parameters
    ----------
    dbpath : string, path to SQLite database file
    set_object : object (either TestSet or TrainSet) which is stored in the database

    Returns
    -------
    return_list : list of tuples containing the name of the feature and the length of the corresponding list or
    1d numpy array
    """
    engine = create_engine('sqlite:////' + dbpath)
    session_cl = sessionmaker(bind=engine)
    session = session_cl()
    return_list = []
    tmp_object = session.query(set_object).get(1)
    for feature in tmp_object.features:
        if type(tmp_object.features[feature]) is np.ndarray:
            flength = tmp_object.features[feature].shape[0]
        else:
            flength = 1
        return_list.append((feature, flength))
    session.close()
    return return_list