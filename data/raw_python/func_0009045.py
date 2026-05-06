def count_children(obj, type=None):
    """Return the number of children of obj, optionally restricting by class"""
    if type is None:
        return len(obj)
    else:
        # there doesn't appear to be any hdf5 function for getting this
        # information without inspecting each child, which makes this somewhat
        # slow
        return sum(1 for x in obj if obj.get(x, getclass=True) is type)