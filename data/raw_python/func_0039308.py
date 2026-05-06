def has_field(mc, field_name):
    """
    detect if a model has a given field has

    :param field_name:
    :param mc:
    :return:
    """
    try:
        mc._meta.get_field(field_name)
    except FieldDoesNotExist:
        return False
    return True