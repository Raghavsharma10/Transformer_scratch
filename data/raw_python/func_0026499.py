def list_users(ctx, search, uuid, active):
    """List all locally known users"""

    users = ctx.obj['db'].objectmodels['user']

    for found_user in users.find():
        if not search or (search and search in found_user.name):
            # TODO: Not 2.x compatible
            print(found_user.name, end=' ' if active or uuid else '\n')
            if uuid:
                print(found_user.uuid, end=' ' if active else '\n')
            if active:
                print(found_user.active)

    log("Done")