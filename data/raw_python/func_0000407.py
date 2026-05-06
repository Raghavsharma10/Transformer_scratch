def create_handler(target: str):
    """Create a handler for logging to ``target``"""
    if target == 'stderr':
        return logging.StreamHandler(sys.stderr)
    elif target == 'stdout':
        return logging.StreamHandler(sys.stdout)
    else:
        return logging.handlers.WatchedFileHandler(filename=target)