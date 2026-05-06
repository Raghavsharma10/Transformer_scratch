def create_user(ctx):
    """Creates a new local user"""

    try:
        new_user = _create_user(ctx)

        new_user.save()
        log("Done")
    except KeyError:
        log('User already exists', lvl=warn)