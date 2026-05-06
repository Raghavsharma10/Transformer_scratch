def init(deb1, deb2=False):
    """Initialize DEBUG and DEBUGALL.

    Allows other modules to set DEBUG and DEBUGALL, so their
    call to dprint or dprintx generate output.

    Args:
        deb1 (bool): value of DEBUG to set
        deb2 (bool): optional - value of DEBUGALL to set,
                     defaults to False.

    """
    global DEBUG        # pylint: disable=global-statement
    global DEBUGALL     # pylint: disable=global-statement
    DEBUG = deb1
    DEBUGALL = deb2