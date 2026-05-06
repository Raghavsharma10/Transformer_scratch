def user(ctx, username, password):
    """[GROUP] User management operations"""

    ctx.obj['username'] = username
    ctx.obj['password'] = password