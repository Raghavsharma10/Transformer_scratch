def get_parent_name(obj):
    """ Gets the name of the object containing @obj and returns as a string

        @obj: any python object

        -> #str parent object name or None
        ..
            from vital.debug import get_parent_name

            get_parent_name(get_parent_name)
            # -> 'vital.debug'

            get_parent_name(vital.debug)
            # -> 'vital'

            get_parent_name(str)
            # -> 'builtins'
        ..
    """
    parent_obj = get_parent_obj(obj)
    parent_name = get_obj_name(parent_obj) if parent_obj else None
    n = 0
    while parent_obj and n < 2500:
        parent_obj = get_parent_obj(parent_obj)
        if parent_obj:
            parent_name = "{}.{}".format(get_obj_name(parent_obj), parent_name)
            n += 1
    if not parent_name or not len(parent_name):
        parent_name = None
        objname = get_obj_name(obj)
        if objname and len(objname.split(".")) > 1:
            return ".".join(objname.split(".")[:-1])
        return None
    return parent_name