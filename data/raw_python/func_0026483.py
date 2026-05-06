def launch(run=True, **args):
    """Bootstrap basics, assemble graph and hand over control to the Core
    component"""

    verbosity['console'] = args['log'] if not args['quiet'] else 100
    verbosity['global'] = min(args['log'], args['logfileverbosity'])
    verbosity['file'] = args['logfileverbosity'] if args['dolog'] else 100
    set_logfile(args['logfilepath'], args['instance'])

    if args['livelog'] is True:
        from hfos import logger
        logger.live = True

    hfoslog("Running with Python", sys.version.replace("\n", ""),
            sys.platform, lvl=debug, emitter='CORE')
    hfoslog("Interpreter executable:", sys.executable, emitter='CORE')
    if args['cert'] is not None:
        hfoslog("Warning! Using SSL without nginx is currently not broken!",
                lvl=critical, emitter='CORE')

    hfoslog("Initializing database access", emitter='CORE', lvl=debug)
    initialize(args['dbhost'], args['dbname'], args['instance'])

    server = construct_graph(args)
    if run and not args['norun']:
        server.run()

    return server