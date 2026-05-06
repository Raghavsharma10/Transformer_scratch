def check_role_permissions(role, permissions, current_permissions):
    """
    Checks the the passed in role (can be user, group or AnonymousUser)  has all the passed
    in permissions, granting them if necessary.
    """
    role_permissions = []

    # get all the current permissions, we'll remove these as we verify they should still be granted
    for permission in permissions:
        splits = permission.split(".")
        if len(splits) != 2 and len(splits) != 3:
            sys.stderr.write("  invalid permission %s, ignoring\n" % permission)
            continue

        app = splits[0]
        codenames = []

        if len(splits) == 2:
            codenames.append(splits[1])
        else:
            (object, action) = splits[1:]

            # if this is a wildcard, then query our database for all the permissions that exist on this object
            if action == '*':
                for perm in Permission.objects.filter(codename__startswith="%s_" % object, content_type__app_label=app):
                    codenames.append(perm.codename)
            # otherwise, this is an error, continue
            else:
                sys.stderr.write("  invalid permission %s, ignoring\n" % permission)
                continue

        if len(codenames) == 0:
            continue

        for codename in codenames:
            # the full codename for this permission
            full_codename = "%s.%s" % (app, codename)

            # this marks all the permissions which should remain
            role_permissions.append(full_codename)

            try:
                assign_perm(full_codename, role)
            except ObjectDoesNotExist:
                pass
                # sys.stderr.write("  unknown permission %s, ignoring\n" % permission)

    # remove any that are extra
    for permission in current_permissions:
        if isinstance(permission, str):
            key = permission
        else:
            key = "%s.%s" % (permission.content_type.app_label, permission.codename)

        if key not in role_permissions:
            remove_perm(key, role)