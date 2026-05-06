def main(doc, tool, dispatch_only=None):
    """gcdt tools parametrized main function to initiate gcdt lifecycle.

    :param doc: docopt string
    :param tool: gcdt tool (gcdt, kumo, tenkai, ramuda, yugen)
    :param dispatch_only: list of commands which do not use gcdt lifecycle
    :return: exit_code
    """
    # Use signal handler to throw exception which can be caught to allow
    # graceful exit.
    # here: https://stackoverflow.com/questions/26414704/how-does-a-python-process-exit-gracefully-after-receiving-sigterm-while-waiting
    signal.signal(signal.SIGTERM, signal_handler)  # Jenkins
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl-C

    try:
        arguments = docopt(doc, sys.argv[1:])
        command = get_command(arguments)
        # DEBUG mode (if requested)
        verbose = arguments.pop('--verbose', False)
        if verbose:
            logging_config['loggers']['gcdt']['level'] = 'DEBUG'
        dictConfig(logging_config)

        if dispatch_only is None:
            dispatch_only = ['version']
        assert tool in ['gcdt', 'kumo', 'tenkai', 'ramuda', 'yugen']

        if command in dispatch_only:
            # handle commands that do not need a lifecycle
            # Note: `dispatch_only` commands do not have a check for ENV variable!
            check_gcdt_update()
            return cmd.dispatch(arguments)
        else:
            env = get_env()
            if not env:
                log.error('\'ENV\' environment variable not set!')
                return 1

            awsclient = AWSClient(botocore.session.get_session())
            return lifecycle(awsclient, env, tool, command, arguments)
    except GracefulExit as e:
        log.info('Received %s signal - exiting command \'%s %s\'',
                 str(e), tool, command)
        return 1