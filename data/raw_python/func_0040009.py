def _value_properties_are_referenced(val):
    """
    val is a dictionary
    :param val:
    :return: True/False
    """
    if ((u'properties' in val.keys()) and
            (u'$ref' in val['properties'].keys())):
        return True
    return False