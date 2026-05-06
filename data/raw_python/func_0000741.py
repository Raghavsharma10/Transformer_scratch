def parse_permissions(perms):
    """ Parse permissions ("perms") which are either exact permission
    names or the keyword 'all'.

    :param perms: List or comma-separated string of nefertari permission
        names, or 'all'
    """
    if isinstance(perms, six.string_types):
        perms = perms.split(',')
    perms = [perm.strip().lower() for perm in perms]
    if 'all' in perms:
        return ALL_PERMISSIONS
    return validate_permissions(perms)