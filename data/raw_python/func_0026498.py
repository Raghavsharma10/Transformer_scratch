def change_password(ctx):
    """Change password of an existing user"""

    username, passhash = _get_credentials(ctx.obj['username'],
                                          ctx.obj['password'],
                                          ctx.obj['db'])

    change_user = ctx.obj['db'].objectmodels['user'].find_one({
        'name': username
    })
    if change_user is None:
        log('No such user', lvl=warn)
        return

    change_user.passhash = passhash
    change_user.save()

    log("Done")