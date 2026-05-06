def build_colorized_logger(
    name=os.getenv(
        "LOG_NAME",
        "client"),
    config="colors-logging.json",
    log_level=logging.INFO,
    log_config_path="{}/logging.json".format(
        os.getenv(
            "LOG_CFG",
            os.path.dirname(os.path.realpath(__file__))))):
    """build_colorized_logger

    :param name: name that shows in the logger
    :param config: name of the config file
    :param log_level: level to log
    :param log_config_path: path to log config file
    """

    override_config = os.getenv(
        "SHARED_LOG_CFG",
        None)
    debug_log_config = bool(os.getenv(
        "DEBUG_SHARED_LOG_CFG",
        "0") == "1")
    if override_config:
        if debug_log_config:
            print((
                "creating logger config env var: "
                "SHARED_LOG_CFG={}".format(
                    override_config)))
        if os.path.exists(override_config):
            setup_logging(
                default_level=log_level,
                default_path=override_config)
            return logging.getLogger(name)
        if debug_log_config:
            print((
                "Failed to find log config using env var: "
                "SHARED_LOG_CFG={}".format(
                    override_config)))
    else:
        if debug_log_config:
            print((
                "Not using shared logging env var: "
                "SHARED_LOG_CFG={}".format(
                    override_config)))
    # allow a shared log config across all components

    use_config = ("{}").format(
        config)

    if not os.path.exists(use_config):
        use_config = ("./antinex_client/log/{}").format(
                            config)
        if not os.path.exists(use_config):
            use_config = log_config_path
            if not os.path.exists(use_config):
                use_config = ("./log/{}").format(
                            config)
                if not os.path.exists(use_config):
                    use_config = ("./antinex_client/log/{}").format(
                                "logging.json")
                # find the last log config backup from the base of the repo
            # find the log config from the defaults with the env LOG_CFG
        # find the log config from the base of the repo
    # find the log config by the given path

    setup_logging(
        default_level=log_level,
        default_path=use_config)

    return logging.getLogger(name)