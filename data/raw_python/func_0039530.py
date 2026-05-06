def return_single_path_base(dbpath, set_object, object_id):
    """
    Generic function which returns a path (path is relative to the path_to_set stored in the database) of an object
    specified by the object_id

    Parameters
    ----------
    dbpath : string, path to SQLite database file
    set_object : object (either TestSet or TrainSet) which is stored in the database
    object_id : int, id of object in database

    Returns
    -------
    path : string
    """
    engine = create_engine('sqlite:////' + dbpath)
    session_cl = sessionmaker(bind=engine)
    session = session_cl()
    tmp_object = session.query(set_object).get(object_id)
    session.close()
    return tmp_object.path