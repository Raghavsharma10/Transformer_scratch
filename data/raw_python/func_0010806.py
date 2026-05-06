def check_all_permissions(sender, **kwargs):
    """
    This syncdb checks our PERMISSIONS setting in settings.py and makes sure all those permissions
    actually exit.
    """
    if not is_permissions_app(sender):
        return

    config = getattr(settings, 'PERMISSIONS', dict())

    # for each of our items
    for natural_key, permissions in config.items():
        # if the natural key '*' then that means add to all objects
        if natural_key == '*':
            # for each of our content types
            for content_type in ContentType.objects.all():
                for permission in permissions:
                    add_permission(content_type, permission)

        # otherwise, this is on a specific content type, add for each of those
        else:
            app, model = natural_key.split('.')
            try:
                content_type = ContentType.objects.get_by_natural_key(app, model)
            except ContentType.DoesNotExist:
                continue

            # add each permission
            for permission in permissions:
                add_permission(content_type, permission)