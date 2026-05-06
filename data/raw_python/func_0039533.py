def return_multiple_convert_numpy_base(dbpath, folder_path, set_object, start_id, end_id, converter, add_args=None):
    """
    Generic function which converts several objects, with ids in the range (start_id, end_id)
    into a 2d numpy array and returns the array, the conversion is done by the 'converter' function

    Parameters
    ----------
    dbpath : string, path to SQLite database file
    folder_path : string, path to folder where the files are stored
    set_object : object (either TestSet or TrainSet) which is stored in the database
    start_id : the id of the first object to be converted
    end_id : the id of the last object to be converted
    converter : function, which takes the path of a data point and *args as parameters and returns a numpy array
    add_args : optional arguments for the converter (list/dictionary/tuple/whatever). if None, the
    converter should take only one input argument - the file path. default value: None

    Returns
    -------
    result : 2-dimensional ndarray
    """
    engine = create_engine('sqlite:////' + dbpath)
    session_cl = sessionmaker(bind=engine)
    session = session_cl()
    tmp_object = session.query(set_object).get(start_id)
    if add_args is None:
        converted = converter(join(folder_path, tmp_object.path))
    else:
        converted = converter(join(folder_path, tmp_object.path), add_args)
    if len(converted.shape) == 0:
        columns_amt = 1
    else:
        columns_amt = converted.shape[0]
    return_array = np.zeros([end_id - start_id + 1, columns_amt])
    for i in xrange(end_id - start_id + 1):
        tmp_object = session.query(set_object).get(start_id + i)
        if add_args is None:
            return_array[i, :] = converter(join(folder_path, tmp_object.path))
        else:
            return_array[i, :] = converter(join(folder_path, tmp_object.path), add_args)
    session.close()
    return return_array