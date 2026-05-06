def reraise(error):
    """Re-raises the error that was processed by prepare_for_reraise earlier."""
    if hasattr(error, "_type_"):
        six.reraise(type(error), error, error._traceback)
    raise error