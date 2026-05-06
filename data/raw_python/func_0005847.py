def compile(conf):
    """Compiles classic uWSGI configuration file using the default
    or given `uwsgiconf` configuration module.

    """
    with errorprint():
        config = ConfModule(conf)
        for conf in config.configurations:
            conf.format(do_print=True)