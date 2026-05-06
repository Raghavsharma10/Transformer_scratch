def version():
    """Output version of gcdt tools and plugins."""
    log.info('gcdt version %s' % __version__)
    tools = get_plugin_versions('gcdttool10')
    if tools:
        log.info('gcdt tools:')
        for p, v in tools.items():
            log.info(' * %s version %s' % (p, v))
    log.info('gcdt plugins:')
    for p, v in get_plugin_versions().items():
        log.info(' * %s version %s' % (p, v))
    generators = get_plugin_versions('gcdtgen10')
    if generators:
        log.info('gcdt scaffolding generators:')
        for p, v in generators.items():
            log.info(' * %s version %s' % (p, v))