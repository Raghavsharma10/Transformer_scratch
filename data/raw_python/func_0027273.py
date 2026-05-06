def get_permissions_description(view, method):
    """
    Returns permissions description in format:
    ### Permissions:
    * permission1 name
     * permission1 docstring
    * permission2 name
     * permission2 docstring
    """
    if not hasattr(view, 'permission_classes'):
        return ''

    description = ''
    for permission_class in view.permission_classes:
        if permission_class == core_permissions.ActionsPermission:
            actions_perm_description = get_actions_permission_description(view, method)
            if actions_perm_description:
                description += '\n' + actions_perm_description if description else actions_perm_description
            continue
        perm_description = get_entity_description(permission_class)
        description += '\n' + perm_description if description else perm_description

    return '### Permissions:\n' + description if description else ''