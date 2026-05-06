def setup(product_name):
    """Setup logging."""
    if CONF.log_config:
        _load_log_config(CONF.log_config)
    else:
        _setup_logging_from_conf()
    sys.excepthook = _create_logging_excepthook(product_name)