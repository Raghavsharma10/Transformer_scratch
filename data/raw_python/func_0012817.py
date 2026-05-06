def _has_name(soup_obj):
    """checks if soup_obj is really a soup object or just a string
    If it has a name it is a soup object"""
    try:
        name = soup_obj.name
        if name == None:
            return False
        return True
    except AttributeError:
        return False