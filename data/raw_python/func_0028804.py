def build_logger(
    name=os.getenv(
        "LOG_NAME",
        "client"),
    config="logging.json",
    log_level=logging.INFO,
    log_config_path="{}/logging.json".format(
        os.getenv(
            "LOG_CFG",
            os.path.dirname(os.path.realpath(__file__))))):
    """build_logger

    :param name: name that shows in the logger
    :param config: name of the config file
    :param log_level: level to log
    :param log_config_path: path to log config file
    """
    use_config = ("./log/{}").format(
                    "{}".format(
                        config))
    if not os.path.exists(use_config):
        use_config = log_config_path
        if not os.path.exists(use_config):
            use_config = ("./antinex_client/log/{}").format(
                            "logging.json")
    # find the log processing

    setup_logging(
        default_level=log_level,
        default_path=use_config)

    return logging.getLogger(name)