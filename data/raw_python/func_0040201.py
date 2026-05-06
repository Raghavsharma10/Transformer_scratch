def action(act, config):
    """
    CLI action preprocessor
    """
    if not config:
        pass
    elif act is "list":
        do_list()
    else:
        config_dir = os.path.join(CONFIG_ROOT, config)
        globals()["do_" + act](config, config_dir)