def enable(ctx):
    """Enable an existing user"""

    if ctx.obj['username'] is None:
        log('Specify the username with "iso db user --username ..."')
        return

    change_user = ctx.obj['db'].objectmodels['user'].find_one({
        'name': ctx.obj['username']
    })

    change_user.active = True
    change_user.save()
    log('Done')