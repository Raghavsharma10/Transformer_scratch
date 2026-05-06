def loado(obj, class_=None):
    """
    Convert a dictionary or a list of dictionaries into a single Physical Information Object or a list of such objects.

    :param obj: Dictionary or list to convert to Physical Information Objects.
    :param class_: Subclass of :class:`.Pio` to produce, if not unambiguous
    :return: Single object derived from :class:`.Pio` or a list of such object.
    """
    if isinstance(obj, list):
        return [_dict_to_pio(i, class_=class_) for i in obj]
    elif isinstance(obj, dict):
        return _dict_to_pio(obj, class_=class_)
    else:
        raise ValueError('expecting list or dictionary as outermost structure')