def equalfield(bunchdt, data, commdct, idfobj1, idfobj2, fieldname, places=7):
    """returns true if the two fields are equal
    will test for retaincase
    places is used if the field is float/real"""
    # TODO test if both objects are of same type
    key1 = idfobj1.obj[0].upper()
    key2 = idfobj2.obj[0].upper()
    if key1 != key2:
        raise NotSameObjectError
    vee2 = idfobj2[fieldname]
    return isfieldvalue(
        bunchdt, data, commdct,
        idfobj1, fieldname, vee2, places=places)