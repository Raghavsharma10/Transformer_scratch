def _create_user(ctx):
    """Internal method to create a normal user"""

    username, passhash = _get_credentials(ctx.obj['username'],
                                          ctx.obj['password'],
                                          ctx.obj['db'])

    if ctx.obj['db'].objectmodels['user'].count({'name': username}) > 0:
        raise KeyError()

    new_user = ctx.obj['db'].objectmodels['user']({
        'uuid': str(uuid4()),
        'created': std_now()
    })

    new_user.name = username
    new_user.passhash = passhash

    return new_user