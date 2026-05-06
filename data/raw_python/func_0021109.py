def change_object_link_card(obj, perms):
    """
    If the user has permission to change `obj`, show a link to its Admin page.
    obj -- An object like Movie, Play, ClassicalWork, Publication, etc.
    perms -- The `perms` object that it's the template.
    """
    # eg: 'movie' or 'classicalwork':
    name = obj.__class__.__name__.lower()
    permission = 'spectator.can_edit_{}'.format(name)
    # eg: 'admin:events_classicalwork_change':
    change_url_name = 'admin:{}_{}_change'.format(obj._meta.app_label, name)

    return {
        'display_link': (permission in perms),
        'change_url': reverse(change_url_name, args=[obj.id])
    }