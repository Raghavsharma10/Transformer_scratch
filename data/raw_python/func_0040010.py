def _value_is_type_text(val):
    """
    val is a dictionary
    :param val:
    :return: True/False
    """
    if ((u'type' in val.keys()) and
            (val['type'].lower() == u"text")):
        return True
    return False