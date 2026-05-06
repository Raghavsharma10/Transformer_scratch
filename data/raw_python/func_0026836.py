def permission_to_pyramid_acls(permissions):
    """
    Returns a list of permissions in a format understood by pyramid
    :param permissions:
    :return:
    """
    acls = []
    for perm in permissions:
        if perm.type == "user":
            acls.append((Allow, perm.user.id, perm.perm_name))
        elif perm.type == "group":
            acls.append((Allow, "group:%s" % perm.group.id, perm.perm_name))
    return acls