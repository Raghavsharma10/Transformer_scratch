def copy_features_base(dbpath_origin, dbpath_destination, set_object, force_copy=False):
    """
    Generic function which copies features from one database to another (base object should be of the same type)

    Parameters
    ----------
    dbpath_origin : string, path to SQLite database file from which the features will be copied
    dbpath_destination : string, path to SQLite database file to which the features will be copied
    set_object : object (either TestSet or TrainSet) which is stored in the database
    force_copy : boolean, if True - will overwrite features with same name when copying, if False, won't;
    default value: False

    Returns
    -------
    None
    """
    engine_origin = create_engine('sqlite:////' + dbpath_origin)
    engine_destination = create_engine('sqlite:////' + dbpath_destination)
    session_cl_origin = sessionmaker(bind=engine_origin)
    session_cl_destination = sessionmaker(bind=engine_destination)
    session_origin = session_cl_origin()
    session_destination = session_cl_destination()
    if force_copy is True:
        for i in session_origin.query(set_object).order_by(set_object.id):
            dest_obj = session_destination.query(set_object).get(i.id)
            for feature in i.features:
                if dest_obj.features is not None:
                    dest_obj.features[feature] = i.features[feature]
                else:
                    dest_obj.features = {feature: i.features[feature]}
    else:
        for i in session_origin.query(set_object).order_by(set_object.id):
            dest_obj = session_destination.query(set_object).get(i.id)
            for feature in i.features:
                if dest_obj.features is not None:
                    if (feature not in dest_obj.features) or force_copy is True:
                        dest_obj.features[feature] = i.features[feature]
                else:
                    dest_obj.features = {feature: i.features[feature]}
    session_origin.close()
    session_destination.commit()
    session_destination.close()
    return None