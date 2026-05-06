def initialize(config=None):
    """Initializes oz"""

    # Load the config file
    if config == None:
        config = {}
        config_source = None

        try:
            with open(os.environ.get("OZ_CONFIG", "config.py")) as f:
                config_source = f.read()
        except Exception:
            tornado.log.gen_log.info("Could not read config.py", exc_info=True)

        if config_source != None:
            tornado.util.exec_in(config_source, config, config)

    # Load the plugins
    for p in config.get("plugins", ["oz.core"]):
        plugin(p)

    # Set the options
    for key, value in config.get("app_options", {}).items():
        setattr(tornado.options.options, key, value)

    # Generate the application settings
    global settings
    settings = tornado.options.options.as_dict()
    settings["ui_modules"] = _uimodules
    settings["project_name"] = config.get("project_name")