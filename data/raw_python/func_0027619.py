def is_error_of_type(exc, ref_type):
    """
    Helper function to determine if some exception is of some type, by also looking at its declared __cause__

    :param exc:
    :param ref_type:
    :return:
    """
    if isinstance(exc, ref_type):
        return True
    elif hasattr(exc, '__cause__') and exc.__cause__ is not None:
        return is_error_of_type(exc.__cause__, ref_type)