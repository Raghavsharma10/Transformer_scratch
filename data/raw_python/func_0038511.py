def instantiate(config):

    """
    instantiate all registered vodka applications

    Args:
        config (dict or MungeConfig): configuration object
    """

    for handle, cfg in list(config["apps"].items()):
        if not cfg.get("enabled", True):
            continue
        app = get_application(handle)
        instances[app.handle] = app(cfg)