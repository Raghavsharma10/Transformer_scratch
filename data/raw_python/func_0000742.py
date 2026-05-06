def parse_acl(acl_string):
    """ Parse raw string :acl_string: of RAML-defined ACLs.

    If :acl_string: is blank or None, all permissions are given.
    Values of ACL action and principal are parsed using `actions` and
    `special_principals` maps and are looked up after `strip()` and
    `lower()`.

    ACEs in :acl_string: may be separated by newlines or semicolons.
    Action, principal and permission lists must be separated by spaces.
    Permissions must be comma-separated.
    E.g. 'allow everyone view,create,update' and 'deny authenticated delete'

    :param acl_string: Raw RAML string containing defined ACEs.
    """
    if not acl_string:
        return [ALLOW_ALL]

    aces_list = acl_string.replace('\n', ';').split(';')
    aces_list = [ace.strip().split(' ', 2) for ace in aces_list if ace]
    aces_list = [(a, b, c.split(',')) for a, b, c in aces_list]
    result_acl = []

    for action_str, princ_str, perms in aces_list:
        # Process action
        action_str = action_str.strip().lower()
        action = actions.get(action_str)
        if action is None:
            raise ValueError(
                'Unknown ACL action: {}. Valid actions: {}'.format(
                    action_str, list(actions.keys())))

        # Process principal
        princ_str = princ_str.strip().lower()
        if princ_str in special_principals:
            principal = special_principals[princ_str]
        elif is_callable_tag(princ_str):
            principal = resolve_to_callable(princ_str)
        else:
            principal = princ_str

        # Process permissions
        permissions = parse_permissions(perms)

        result_acl.append((action, principal, permissions))

    return result_acl