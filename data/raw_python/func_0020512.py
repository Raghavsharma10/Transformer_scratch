def wrap_name_from_git(prefix, suffix, *args, **kwargs):
    """
    wraps the result of make_name_from_git in a suffix and postfix
    adding separators for each.

    see docstring for make_name_from_git for a full list of parameters
    """
    # 64 is maximum length allowed by OpenShift
    # 2 is the number of dashes that will be added
    prefix = ''.join(filter(VALID_BUILD_CONFIG_NAME_CHARS.match, list(prefix)))
    suffix = ''.join(filter(VALID_BUILD_CONFIG_NAME_CHARS.match, list(suffix)))
    kwargs['limit'] = kwargs.get('limit', 64) - len(prefix) - len(suffix) - 2
    name_from_git = make_name_from_git(*args, **kwargs)
    return '-'.join([prefix, name_from_git, suffix])