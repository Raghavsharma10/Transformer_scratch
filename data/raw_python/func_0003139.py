def set_logging( logfilename=None, level=None ):
    """
    Set a logging configuration, with a rolling file appender.
    If passed a filename, use it as the logfile, else use a default name.

    The default logfile is \c sparqlkernel.log, placed in the directory given
    by (in that order) the \c LOGDIR environment variable, the logdir
    specified upon kernel installation or the default temporal directory.
    """
    if logfilename is None:
        # Find the logging diectory
        logdir = os.environ.get( 'LOGDIR' )
        if logdir is None:
            logdir = os.environ.get( 'LOGDIR_DEFAULT', tempfile.gettempdir() )
        # Define the log filename
        basename = __name__.split('.')[-2]
        logfilename = os.path.join( logdir, basename + '.log' )
    LOGCONFIG['handlers']['default']['filename'] = logfilename

    if level is not None:
        LOGCONFIG['loggers']['sparqlkernel']['level'] = level

    dictConfig( LOGCONFIG )