def set_env_value(obj, attribute, value):
    """
    Set the environment variable value for the attribute of the
    given object.

    For example, `set_env_value(predix.security.uaa, 'uri', 'http://...')`
    will set the environment variable PREDIX_SECURITY_UAA_URI to the given
    uri.
    """
    varname = get_env_key(obj, attribute)
    os.environ[varname] = value
    return varname