def get_actions_permission_description(view, method):
    """
    Returns actions permissions description in format:
    * permission1 name
     * permission1 docstring
    * permission2 name
     * permission2 docstring
    """
    action = getattr(view, 'action', None)
    if action is None:
        return ''

    if hasattr(view, action + '_permissions'):
        permission_types = (action,)
    elif method in SAFE_METHODS:
        permission_types = ('safe_methods', '%s_extra' % action)
    else:
        permission_types = ('unsafe_methods', '%s_extra' % action)

    description = ''
    for permission_type in permission_types:
        action_perms = getattr(view, permission_type + '_permissions', [])
        for permission in action_perms:
            action_perm_description = get_entity_description(permission)
            description += '\n' + action_perm_description if description else action_perm_description

    return description