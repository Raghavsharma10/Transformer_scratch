def get_remote_host_environment_variable(host, environment_variable):
    """Retrieves the value of an environment variable of a
    remote host over SSH

    :param host: (str) host to query
    :param environment_variable: (str) variable to query
    :return: (str) value of the environment variable
    :raises: TypeError, CommandError
    """
    log = logging.getLogger(mod_logger + '.get_remote_host_environment_variable')
    if not isinstance(host, basestring):
        msg = 'host argument must be a string'
        log.error(msg)
        raise TypeError(msg)
    if not isinstance(environment_variable, basestring):
        msg = 'environment_variable argument must be a string'
        log.error(msg)
        raise TypeError(msg)
    log.info('Checking host {h} for environment variable: {v}...'.format(h=host, v=environment_variable))
    command = ['ssh', '{h}'.format(h=host), 'echo ${v}'.format(v=environment_variable)]
    try:
        result = run_command(command, timeout_sec=5.0)
        code = result['code']
    except CommandError:
        raise
    if code != 0:
        msg = 'There was a problem checking the remote host {h} over SSH, return code: {c}'.format(
                h=host, c=code)
        log.error(msg)
        raise CommandError(msg)
    else:
        value = result['output'].strip()
        log.info('Environment variable {e} on host {h} value is: {v}'.format(
                e=environment_variable, h=host, v=value))
    return value