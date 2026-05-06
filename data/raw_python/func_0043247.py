def install(args):
    " Install site from sources or module "

    # Deactivate virtualenv
    if 'VIRTUAL_ENV' in environ:
        LOGGER.warning('Virtualenv enabled: %s' % environ['VIRTUAL_ENV'])

    # Install from base modules
    if args.module:
        args.src = op.join(settings.MOD_DIR, args.module)
        assert op.exists(args.src), "Not found module: %s" % args.module

    # Fix project name
    args.PROJECT = args.PROJECT.replace('-', '_')

    args.home = op.abspath(args.path)

    # Create engine
    engine = Installer(args)
    args.deploy_dir = engine.target_dir

    # Check dir exists
    assert args.info or args.repeat or args.update or not op.exists(
        engine.target_dir), "Path %s exists. Stop deploy." % args.deploy_dir

    try:
        if args.repeat:
            site = Site(engine.target_dir)
            site.run_install()
            return site

        site = engine.clone_source()
        if not site:
            return True

        engine.build(args.update)
        site.run_install()
        return site

    except (CalledProcessError, AssertionError):
        LOGGER.error("Installation failed")
        LOGGER.error("Fix errors and repeat installation with (-r) or run 'makesite uninstall %s' for cancel." % args.deploy_dir)
        raise