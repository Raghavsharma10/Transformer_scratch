def set_attr(obj, path, value):
    """
    SAME AS object.__setattr__(), BUT USES DOT-DELIMITED path
    RETURN OLD VALUE
    """
    try:
        return _set_attr(obj, split_field(path), value)
    except Exception as e:
        Log = get_logger()
        if PATH_NOT_FOUND in e:
            Log.warning(PATH_NOT_FOUND + ": {{path}}", path=path, cause=e)
        else:
            Log.error("Problem setting value", cause=e)