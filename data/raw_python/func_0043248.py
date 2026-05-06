def console():
    " Enter point "
    autocomplete()
    config = settings.MakesiteParser()
    config.read([
        settings.BASECONFIG, settings.HOMECONFIG,
        op.join(settings.MAKESITE_HOME or '', settings.CFGNAME),
        op.join(op.curdir, settings.CFGNAME),
    ])
    argv = []
    alias = dict(config.items('alias'))
    names = alias.keys()
    for arg in sys.argv[1:]:
        if arg in names:
            argv += alias[arg].split()
            continue
        argv.append(arg)

    main(argv)