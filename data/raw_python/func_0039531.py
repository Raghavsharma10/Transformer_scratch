def return_single_features_base(dbpath, set_object, object_id):
    """
    Generic function which returns the features of an object specified by the object_id

    Parameters
    ----------
    dbpath : string, path to SQLite database file
    set_object : object (either TestSet or TrainSet) which is stored in the database
    object_id : int, id of object in database

    Returns
    -------
    features : dict containing the features
    """
    engine = create_engine('sqlite:////' + dbpath)
    session_cl = sessionmaker(bind=engine)
    session = session_cl()
    tmp_object = session.query(set_object).get(object_id)
    session.close()
    return tmp_object.features