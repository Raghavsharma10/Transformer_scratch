def validate_permissions(perms):
    """ Validate :perms: contains valid permissions.

    :param perms: List of permission names or ALL_PERMISSIONS.
    """
    if not isinstance(perms, (list, tuple)):
        perms = [perms]
    valid_perms = set(PERMISSIONS.values())
    if ALL_PERMISSIONS in perms:
        return perms
    if set(perms) - valid_perms:
        raise ValueError(
            'Invalid ACL permission names. Valid permissions '
            'are: {}'.format(', '.join(valid_perms)))
    return perms