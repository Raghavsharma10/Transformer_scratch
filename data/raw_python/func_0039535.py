def delete_feature_base(dbpath, set_object, name):
    """
    Generic function which deletes a feature from a database

    Parameters
    ----------
    dbpath : string, path to SQLite database file
    set_object : object (either TestSet or TrainSet) which is stored in the database
    name : string, name of the feature to be deleted

    Returns
    -------
    None
    """
    engine = create_engine('sqlite:////' + dbpath)
    session_cl = sessionmaker(bind=engine)
    session = session_cl()
    tmp_object = session.query(set_object).get(1)
    if tmp_object.features is not None and name in tmp_object.features:
        for i in session.query(set_object).order_by(set_object.id):
            del i.features[name]
    session.commit()
    session.close()
    return None