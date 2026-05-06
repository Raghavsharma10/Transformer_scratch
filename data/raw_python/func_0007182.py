def _build_install_args(options):
    '''
    Build the arguments to 'python setup.py install' on the setuptools package
    '''
    install_args = []
    if options.user_install:
        if sys.version_info < (2, 6):
            log.warn('--user requires Python 2.6 or later')
            raise SystemExit(1)
        install_args.append('--user')
    return install_args