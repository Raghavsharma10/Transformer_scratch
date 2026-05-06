def is_jsonable(obj) -> bool:
    """
    Check if an object is jsonable.

    An object is jsonable if it is json serialisable and by loading its json representation the same object is recovered.
    Parameters
    ----------
    obj : 
        Python object

    Returns
    -------
    bool

    >>> is_jsonable([1,2,3])
    True
    >>> is_jsonable((1,2,3))
    False
    >>> is_jsonable({'a':True,'b':1,'c':None})
    True
    """
    try:
        return obj==json.loads(json.dumps(obj))
    except TypeError:
        return False
    except:
        raise