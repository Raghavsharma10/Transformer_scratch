def __objecthasfields(bunchdt, data, commdct, idfobject, places=7, **kwargs):
    """test if the idf object has the field values in kwargs"""
    for key, value in list(kwargs.items()):
        if not isfieldvalue(
                bunchdt, data, commdct,
                idfobject, key, value, places=places):
            return False
    return True