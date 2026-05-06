def remove_properties_containing_None(properties_dict):
    """
    removes keys from a dict those values == None
    json schema validation might fail if they are set and
    the type or format of the property does not match
    """
    # remove empty properties - as validations may fail
    new_dict  = dict()
    for key in properties_dict.keys():
        value = properties_dict[key]
        if value is not None:
            new_dict[key] = value
    return new_dict