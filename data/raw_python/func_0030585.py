def init_library(database_dsn, accounts_password, limited_run = False):
    """Child initializer, setup in Library.process_pool"""

    import os
    import signal

    # Have the child processes ignore the keyboard interrupt, and other signals. Instead, the parent will
    # catch these, and clean up the children.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    #signal.signal(signal.SIGTERM, sigterm_handler)

    os.environ['AMBRY_DB'] = database_dsn
    if accounts_password:
        os.environ['AMBRY_PASSWORD'] = accounts_password
    os.environ['AMBRY_LIMITED_RUN'] = '1' if limited_run else '0'