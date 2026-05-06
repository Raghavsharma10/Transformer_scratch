def setup_daemon_log_file(cfstore):
    """
    Attach file handler to RASH logger.

    :type cfstore: rash.config.ConfigStore

    """
    level = loglevel(cfstore.daemon_log_level)
    handler = logging.FileHandler(filename=cfstore.daemon_log_path)
    handler.setLevel(level)
    logger.setLevel(level)
    logger.addHandler(handler)