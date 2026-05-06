def add_role(ctx, role):
    """Grant a role to an existing user"""

    if role is None:
        log('Specify the role with --role')
        return
    if ctx.obj['username'] is None:
        log('Specify the username with --username')
        return

    change_user = ctx.obj['db'].objectmodels['user'].find_one({
        'name': ctx.obj['username']
    })
    if role not in change_user.roles:
        change_user.roles.append(role)
        change_user.save()
        log('Done')
    else:
        log('User already has that role!', lvl=warn)