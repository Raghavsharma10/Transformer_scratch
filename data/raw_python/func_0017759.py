def init(opts, no_install=False, quiet=False):
    """Initialize a purged environment or copied environment directory

Usage:
  datacats init [-in] [--syslog] [-s NAME] [--address=IP] [--interactive]
                [--site-url SITE_URL] [ENVIRONMENT_DIR [PORT]] [--no-init-db]

Options:
  --address=IP            Address to listen on (Linux-only)
  --interactive           Don't detach from the web container
  -i --image-only         Create the environment but don't start containers
  --no-init-db            Don't initialize the database. Useful for importing other CKANs
  -n --no-sysadmin        Don't prompt for an initial sysadmin user account
  -s --site=NAME          Pick a site to initialize [default: primary]
  --site-url SITE_URL     The site_url to use in API responses (e.g. http://example.org:{port}/)
  --syslog                Log to the syslog

ENVIRONMENT_DIR is an existing datacats environment directory. Defaults to '.'
"""
    if opts['--address'] and is_boot2docker():
        raise DatacatsError('Cannot specify address on boot2docker.')
    environment_dir = opts['ENVIRONMENT_DIR']
    port = opts['PORT']
    address = opts['--address']
    start_web = not opts['--image-only']
    create_sysadmin = not opts['--no-sysadmin']
    site_name = opts['--site']
    site_url = opts['--site-url']
    interactive = opts['--interactive']
    init_db = not opts['--no-init-db']

    environment_dir = abspath(environment_dir or '.')
    log_syslog = opts['--syslog']

    environment = Environment.load(environment_dir, site_name)
    if address:
        environment.address = address
    if port:
        environment.port = int(port)
    if site_url:
        environment.site_url = site_url

    try:
        if environment.sites and site_name in environment.sites:
            raise DatacatsError('Site named {0} already exists.'
                                .format(site_name))
        # There are a couple of steps we can/must skip if we're making a sub-site only
        making_full_environment = not environment.data_exists()

        if not quiet:
            write('Creating environment {0}/{1} '
                  'from existing environment directory "{0}"'
                  .format(environment.name, environment.site_name))
        steps = [
            lambda: environment.create_directories(create_project_dir=False)] + ([
             environment.save,
             environment.create_virtualenv
             ] if making_full_environment else []) + [
                 environment.save_site,
                 environment.start_supporting_containers,
                 environment.fix_storage_permissions,
            ]

        for fn in steps:
            fn()
            if not quiet:
                write('.')
        if not quiet:
            write('\n')
    except:
        if not quiet:
            print
        raise

    return finish_init(environment, start_web, create_sysadmin,
                       log_syslog=log_syslog, do_install=not no_install,
                       quiet=quiet, site_url=site_url, interactive=interactive,
                       init_db=init_db)