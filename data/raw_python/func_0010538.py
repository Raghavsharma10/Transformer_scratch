def logging_stream_install(loglevel):
    """
    Install logger that will output to stderr. If this function ha already installed a handler, replace it.
    :param loglevel: log level for the stream
    """
    formatter = logging.Formatter(LOGGING_FORMAT)
    logger = logging.getLogger()

    logger.removeHandler(LOGGING_HANDLERS['stream'])

    if loglevel == LOGGING_LOGNOTHING:
        streamHandler = None
    else:
        streamHandler = logging.StreamHandler()
        streamHandler.setLevel(loglevel)
        streamHandler.setFormatter(formatter)

    LOGGING_HANDLERS['stream'] = streamHandler

    if streamHandler:
        logger.addHandler(streamHandler)