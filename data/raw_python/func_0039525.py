def return_real_id_base(dbpath, set_object):
    """
    Generic function which returns a list of real_id's

    Parameters
    ----------
    dbpath : string, path to SQLite database file
    set_object : object (either TestSet or TrainSet) which is stored in the database

    Returns
    -------
    return_list : list of real_id values for the dataset (a real_id is the filename minus the suffix and prefix)
    """
    engine = create_engine('sqlite:////' + dbpath)
    session_cl = sessionmaker(bind=engine)
    session = session_cl()
    return_list = []
    for i in session.query(set_object).order_by(set_object.id):
        return_list.append(i.real_id)
    session.close()
    return return_list