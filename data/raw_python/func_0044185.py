def cli_frontend(ctx, verbose):
    """
    Boussole is a commandline interface to build Sass projects using libsass.

    Every project will need a settings file containing all needed settings to
    build it.
    """
    printout = True
    if verbose == 0:
        verbose = 1
        printout = False

    # Verbosity is the inverse of logging levels
    levels = [item for item in BOUSSOLE_LOGGER_CONF]
    levels.reverse()
    # Init the logger config
    root_logger = init_logger(levels[verbose], printout=printout)

    # Init the default context that will be passed to commands
    ctx.obj = {
        'verbosity': verbose,
        'logger': root_logger,
    }