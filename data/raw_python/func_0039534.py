def dump_feature_base(dbpath, set_object, points_amt, feature_name, feature, force_extraction=True):
    """
    Generic function which dumps a list of lists or ndarray of features into database (allows to
    copy features from a pre-existing .txt/.csv/.whatever file, for example)

    Parameters
    ----------
    dbpath : string, path to SQLite database file
    set_object : object (either TestSet or TrainSet) which is stored in the database
    points_amt : int, number of data points in the database
    feature : list of lists or ndarray, contains the data to be written to the database
    force_extraction : boolean, if True - will overwrite any existing feature with this name
    default value: False

    Returns
    -------
    None
    """
    engine = create_engine('sqlite:////' + dbpath)
    session_cl = sessionmaker(bind=engine)
    session = session_cl()
    a = 0

    tmp_object = session.query(set_object).get(1)
    if type(feature) is np.ndarray:
        if feature.shape[0] != points_amt:
            raise errors.WrongSize(feature_name)
        else:
            if tmp_object.features is None:
                for i in session.query(set_object).order_by(set_object.id):
                    i.features = {feature_name: feature_name[a, :]}
                    a += 1
            elif (feature_name not in tmp_object.features) or force_extraction is True:
                for i in session.query(set_object).order_by(set_object.id):
                    i.features[feature_name] = feature_name[a, :]
                    a += 1
    else:
        if len(feature) != points_amt:
            raise errors.WrongSize(feature_name)
        else:
            if tmp_object.features is None:
                for i in session.query(set_object).order_by(set_object.id):
                    i.features = {feature_name: feature_name[a]}
                    a += 1
            elif (feature_name not in tmp_object.features) or force_extraction is True:
                for i in session.query(set_object).order_by(set_object.id):
                    i.features[feature_name] = feature_name[a]
                    a += 1
    session.commit()
    session.close()
    return None