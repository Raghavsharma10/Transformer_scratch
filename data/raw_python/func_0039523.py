def return_features_base(dbpath, set_object, names):
    """
    Generic function which returns a list of extracted features from the database

    Parameters
    ----------
    dbpath : string, path to SQLite database file
    set_object : object (either TestSet or TrainSet) which is stored in the database
    names : list of strings, a list of feature names which are to be retrieved from the database, if equal to 'all',
    all features will be returned

    Returns
    -------
    return_list : list of lists, each 'inside list' corresponds to a single data point, each element of the 'inside
    list' is a feature (can be of any type)
    """
    engine = create_engine('sqlite:////' + dbpath)
    session_cl = sessionmaker(bind=engine)
    session = session_cl()
    return_list = []
    if names == 'all':
        for i in session.query(set_object).order_by(set_object.id):
            row_list = []
            for feature in i.features:
                row_list.append(i.features[feature])
            return_list.append(row_list[:])
    else:
        for i in session.query(set_object).order_by(set_object.id):
            row_list = []
            for feature in i.features:
                if feature in names:
                    row_list.append(i.features[feature])
            return_list.append(row_list[:])
    return return_list