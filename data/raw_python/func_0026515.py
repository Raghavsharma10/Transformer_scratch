def change_owner(ctx, owner, uuid):
    """Changes the ownership of objects"""

    objects = ctx.obj['objects']
    database = ctx.obj['db']

    if uuid is True:
        owner_filter = {'uuid': owner}
    else:
        owner_filter = {'name': owner}

    owner = database.objectmodels['user'].find_one(owner_filter)
    if owner is None:
        log('User unknown.', lvl=error)
        return

    for item in objects:
        item.owner = owner.uuid
        item.save()

    log('Done')