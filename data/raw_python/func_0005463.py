def service_network_restart():
    """Restarts the network service on linux
    :return: None
    :raises CommandError
    """
    log = logging.getLogger(mod_logger + '.service_network_restart')
    command = ['service', 'network', 'restart']
    time.sleep(5)
    try:
        result = run_command(command)
        time.sleep(5)
        code = result['code']
    except CommandError:
        raise
    log.info('Network restart produced output:\n{o}'.format(o=result['output']))

    if code != 0:
        msg = 'Network services did not restart cleanly, exited with code: {c}'.format(c=code)
        log.error(msg)
        raise CommandError(msg)
    else:
        log.info('Successfully restarted networking!')