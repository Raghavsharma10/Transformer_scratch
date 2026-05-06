def get_attr(obj, path):
    """
    SAME AS object.__getattr__(), BUT USES DOT-DELIMITED path
    """
    try:
        return _get_attr(obj, split_field(path))
    except Exception as e:
        Log = get_logger()
        if PATH_NOT_FOUND in e:
            Log.error(PATH_NOT_FOUND+": {{path}}",  path=path, cause=e)
        else:
            Log.error("Problem setting value", e)