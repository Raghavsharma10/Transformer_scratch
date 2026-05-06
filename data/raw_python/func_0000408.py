def initialise_logging(level: str, target: str, short_format: bool):
    """Initialise basic logging facilities"""
    try:
        log_level = getattr(logging, level)
    except AttributeError:
        raise SystemExit(
            "invalid log level %r, expected any of 'DEBUG', 'INFO', 'WARNING', 'ERROR' or 'CRITICAL'" % level
        )
    handler = create_handler(target=target)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)-15s (%(process)d) %(message)s' if not short_format else '%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[handler]
    )