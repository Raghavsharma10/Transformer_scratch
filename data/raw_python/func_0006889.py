def read(alias_name, allow_none=False):
    """Get the raw docker link value.

    Get the raw environment variable for the docker link

    Args:
        alias_name: The environment variable name
        default: The default value if the link isn't available
        allow_none: If the return value can be `None` (i.e. optional)
    """
    warnings.warn('Will be removed in v1.0', DeprecationWarning, stacklevel=2)
    return core.read('{0}_PORT'.format(alias_name), default=None, allow_none=allow_none)