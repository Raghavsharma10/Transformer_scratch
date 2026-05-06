def install(environment, opts):
    """Install or reinstall Python packages within this environment

Usage:
  datacats install [-q] [--address=IP] [ENVIRONMENT [PACKAGE ...]]
  datacats install -c [q] [--address=IP] [ENVIRONMENT]

Options:
  --address=IP          The address to bind to when reloading after install
  -c --clean            Reinstall packages into a clean virtualenv
  -q --quiet            Do not show output from installing packages and requirements.

ENVIRONMENT may be an environment name or a path to an environment directory.
Default: '.'
"""
    environment.require_data()
    install_all(environment, opts['--clean'], verbose=not opts['--quiet'],
        packages=opts['PACKAGE'])

    for site in environment.sites:
        environment = Environment.load(environment.name, site)
        if 'web' in environment.containers_running():
            # FIXME: reload without changing debug setting?
            manage.reload_(environment, {
                '--address': opts['--address'],
                '--background': False,
                '--no-watch': False,
                '--production': False,
                'PORT': None,
                '--syslog': False,
                '--site-url': None,
                '--interactive': False
                })