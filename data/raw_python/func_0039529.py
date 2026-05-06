def return_single_real_id_base(dbpath, set_object, object_id):
    """
    Generic function which returns a real_id string of an object specified by the object_id

    Parameters
    ----------
    dbpath : string, path to SQLite database file
    set_object : object (either TestSet or TrainSet) which is stored in the database
    object_id : int, id of object in database

    Returns
    -------
    real_id : string
    """
    engine = create_engine('sqlite:////' + dbpath)
    session_cl = sessionmaker(bind=engine)
    session = session_cl()
    tmp_object = session.query(set_object).get(object_id)
    session.close()
    return tmp_object.real_id