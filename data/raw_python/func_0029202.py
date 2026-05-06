def signal_handler(signum, frame):
    """
    handle signals.
    example: 'signal.signal(signal.SIGTERM, signal_handler)'
    """
    # signals are CONSTANTS so there is no mapping from signum to description
    # so please add to the mapping in case you use more signals!
    description = '%d' % signum
    if signum == 2:
        description = 'SIGINT'
    elif signum == 15:
        description = 'SIGTERM'
    raise GracefulExit(description)