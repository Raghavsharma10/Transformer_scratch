def cli(ctx):
    """ This is a command line app to get useful stats from a trello board
        and report on them in useful ways.

        Requires the following environment varilables:

        TRELLOSTATS_APP_KEY=<your key here>
        TRELLOSTATS_APP_TOKEN=<your token here>
    """
    ctx.obj = dict()
    ctx.obj['app_key'] = os.environ.get('TRELLOSTATS_APP_KEY')
    ctx.obj['app_token'] = os.environ.get('TRELLOSTATS_APP_TOKEN')
    init_db(db_proxy)