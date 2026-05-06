def _parse_command_line_arguments():
    """ Transform vispy specific command line args to vispy config.
    Put into a function so that any variables dont leak in the vispy namespace.
    """
    global config
    # Get command line args for vispy
    argnames = ['vispy-backend=', 'vispy-gl-debug', 'vispy-glir-file=',
                'vispy-log=', 'vispy-help', 'vispy-profile=', 'vispy-cprofile',
                'vispy-dpi=', 'vispy-audit-tests']
    try:
        opts, args = getopt.getopt(sys.argv[1:], '', argnames)
    except getopt.GetoptError:
        opts = []
    # Use them to set the config values
    for o, a in opts:
        if o.startswith('--vispy'):
            if o == '--vispy-backend':
                config['default_backend'] = a
                logger.info('vispy backend: %s', a)
            elif o == '--vispy-gl-debug':
                config['gl_debug'] = True
            elif o == '--vispy-glir-file':
                config['glir_file'] = a
            elif o == '--vispy-log':
                if ',' in a:
                    verbose, match = a.split(',')
                else:
                    verbose = a
                    match = None
                config['logging_level'] = a
                set_log_level(verbose, match)
            elif o == '--vispy-profile':
                config['profile'] = a
            elif o == '--vispy-cprofile':
                _enable_profiling()
            elif o == '--vispy-help':
                print(VISPY_HELP)
            elif o == '--vispy-dpi':
                config['dpi'] = int(a)
            elif o == '--vispy-audit-tests':
                config['audit_tests'] = True
            else:
                logger.warning("Unsupported vispy flag: %s" % o)