def return_feature_list_base(dbpath, set_object):
    """
    Generic function which returns a list of the names of all available features

    Parameters
    ----------
    dbpath : string, path to SQLite database file
    set_object : object (either TestSet or TrainSet) which is stored in the database

    Returns
    -------
    return_list : list of strings corresponding to all available features
    """
    engine = create_engine('sqlite:////' + dbpath)
    session_cl = sessionmaker(bind=engine)
    session = session_cl()
    return_list = []
    tmp_object = session.query(set_object).get(1)
    for feature in tmp_object.features:
        return_list.append(feature)
    session.close()
    return return_list