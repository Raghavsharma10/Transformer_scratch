def set_attributes(obj, additional_data):
    """
    Given an object and a dictionary, give the object new attributes from that dictionary.

    Uses _strip_column_name to git rid of whitespace/uppercase/special characters.
    """
    for key, value in additional_data.items():
        if hasattr(obj, key):
            raise ValueError("Key %s in additional_data already exists in this object" % key)
        setattr(obj, _strip_column_name(key), value)