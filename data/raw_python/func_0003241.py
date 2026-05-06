def default_start():
    """
    Use `sys.argv` for starting parameters. This is the entry-point of `vlcp-start`
    """
    (config, daemon, pidfile, startup, fork) = parsearg()
    if config is None:
        if os.path.isfile('/etc/vlcp.conf'):
            config = '/etc/vlcp.conf'
        else:
            print('/etc/vlcp.conf is not found; start without configurations.')
    elif not config:
        config = None
    main(config, startup, daemon, pidfile, fork)