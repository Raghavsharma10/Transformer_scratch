def return_single_convert_numpy_base(dbpath, folder_path, set_object, object_id, converter, add_args=None):
    """
    Generic function which converts an object specified by the object_id into a numpy array and returns the array,
    the conversion is done by the 'converter' function

    Parameters
    ----------
    dbpath : string, path to SQLite database file
    folder_path : string, path to folder where the files are stored
    set_object : object (either TestSet or TrainSet) which is stored in the database
    object_id : int, id of object in database
    converter : function, which takes the path of a data point and *args as parameters and returns a numpy array
    add_args : optional arguments for the converter (list/dictionary/tuple/whatever). if None, the
    converter should take only one input argument - the file path. default value: None

    Returns
    -------
    result : ndarray
    """
    engine = create_engine('sqlite:////' + dbpath)
    session_cl = sessionmaker(bind=engine)
    session = session_cl()
    tmp_object = session.query(set_object).get(object_id)
    session.close()
    if add_args is None:
        return converter(join(folder_path, tmp_object.path))
    else:
        return converter(join(folder_path, tmp_object.path), add_args)