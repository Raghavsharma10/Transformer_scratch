def empty(val):
    """
    Checks if value is empty.
    All unknown data types considered as empty values.
    @return: bool
    """
    if val == None:
        return True

    if isinstance(val,str) and len(val) > 0:
        return False

    return True