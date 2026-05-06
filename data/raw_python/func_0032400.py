def init(track_log_handler):
    """
    (Re)initialize track's file handler for track package logger.

    Adds a stdout-printing handler automatically.
    """

    logger = logging.getLogger(__package__)

    # TODO (just document prominently)
    # assume only one trial can run at once right now
    # multi-concurrent-trial support will require complex filter logic
    # based on the currently-running trial (maybe we shouldn't allow multiple
    # trials on different python threads, that's dumb)
    to_rm = [h for h in logger.handlers if isinstance(h, TrackLogHandler)]
    for h in to_rm:
        logger.removeHandler(h)

    if not any(isinstance(h, StdoutHandler) for h in logger.handlers):
        handler = StdoutHandler()
        handler.setFormatter(_FORMATTER)
        logger.addHandler(handler)

    track_log_handler.setFormatter(_FORMATTER)
    logger.addHandler(track_log_handler)

    logger.propagate = False
    logger.setLevel(logging.DEBUG)