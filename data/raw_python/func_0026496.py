def create_admin(ctx):
    """Creates a new local user and assigns admin role"""

    try:
        admin = _create_user(ctx)
        admin.roles.append('admin')

        admin.save()
        log("Done")
    except KeyError:
        log('User already exists', lvl=warn)