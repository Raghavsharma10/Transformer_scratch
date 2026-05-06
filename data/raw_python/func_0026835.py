def permission_to_04_acls(permissions):
    """
    Legacy acl format kept for bw. compatibility
    :param permissions:
    :return:
    """
    acls = []
    for perm in permissions:
        if perm.type == "user":
            acls.append((perm.user.id, perm.perm_name))
        elif perm.type == "group":
            acls.append(("group:%s" % perm.group.id, perm.perm_name))
    return acls