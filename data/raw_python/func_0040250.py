def rebuild(args):
    """Rebuild a target and deps, even if it has been built and cached."""
    if len(args) != 1:
        log.fatal('One target required.')
        app.quit(1)

    app.set_option('disable_cache_fetch', True)
    Butcher.options['cache_fetch'] = False
    build(args)