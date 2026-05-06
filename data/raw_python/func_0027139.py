def prepare_for_reraise(error, exc_info=None):
    """Prepares the exception for re-raising with reraise method.

    This method attaches type and traceback info to the error object
    so that reraise can properly reraise it using this info.

    """
    if not hasattr(error, "_type_"):
        if exc_info is None:
            exc_info = sys.exc_info()
        error._type_ = exc_info[0]
        error._traceback = exc_info[2]
    return error