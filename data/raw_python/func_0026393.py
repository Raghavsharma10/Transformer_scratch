def db(ctx):
    """[GROUP] Database management operations"""

    from hfos import database
    database.initialize(ctx.obj['dbhost'], ctx.obj['dbname'])
    ctx.obj['db'] = database