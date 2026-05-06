def check_all_group_permissions(sender, **kwargs):
    """
    Checks that all the permissions specified in our settings.py are set for our groups.
    """
    if not is_permissions_app(sender):
        return

    config = getattr(settings, 'GROUP_PERMISSIONS', dict())

    # for each of our items
    for name, permissions in config.items():
        # get or create the group
        (group, created) = Group.objects.get_or_create(name=name)
        if created:
            pass

        check_role_permissions(group, permissions, group.permissions.all())