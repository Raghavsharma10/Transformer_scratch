def authInsert(user, role, group, site):
    """
    Authorization function for general insert  
    """
    if not role: return True
    for k, v in user['roles'].iteritems():
        for g in v['group']:
            if k in role.get(g, '').split(':'):
                return True
    return False