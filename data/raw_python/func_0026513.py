def add_action_role(ctx):
    """Adds a role to an action on objects"""

    objects = ctx.obj['objects']
    action = ctx.obj['action']
    role = ctx.obj['role']

    if action is None or role is None:
        log('You need to specify an action or role to the RBAC command group for this to work.', lvl=warn)
        return

    for item in objects:
        if role not in item.perms[action]:
            item.perms[action].append(role)
            item.save()

    log("Done")