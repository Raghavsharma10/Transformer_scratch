def cli(sock, configs, modules, files, log, debug):
    """The CLI."""

    setup_logging(log, debug)

    config = join_configs(configs)

    # load python modules
    load_modules(modules)

    # load python files
    load_files(files)

    # summarize active events and callbacks
    summarize_events()

    gloop = gevent.Greenlet.spawn(loop, sock=sock, config=config)
    gloop.start()
    gloop.join()