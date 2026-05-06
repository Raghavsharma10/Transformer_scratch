def toDict(datastorage_obj, recursive=True):
    """ convert a DataStorage object to a dictionary (useful for saving); it should work for other objects too 
    """
    # if not a DataStorage, convert to it first
    if "items" not in dir(datastorage_obj): datastorage_obj = DataStorage(datastorage_obj)
    return _toDict(datastorage_obj)