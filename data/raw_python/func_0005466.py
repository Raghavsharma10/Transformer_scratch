def run_remote_command(host, command, timeout_sec=5.0):
    """Retrieves the value of an environment variable of a
    remote host over SSH

    :param host: (str) host to query
    :param command: (str) command
    :param timeout_sec (float) seconds to wait before killing the command.
    :return: (str) command output
    :raises: TypeError, CommandError
    """
    log = logging.getLogger(mod_logger + '.run_remote_command')
    if not isinstance(host, basestring):
        msg = 'host argument must be a string'
        raise TypeError(msg)
    if not isinstance(command, basestring):
        msg = 'command argument must be a string'
        raise TypeError(msg)
    log.debug('Running remote command on host: {h}: {c}...'.format(h=host, c=command))
    command = ['ssh', '{h}'.format(h=host), '{c}'.format(c=command)]
    try:
        result = run_command(command, timeout_sec=timeout_sec)
        code = result['code']
    except CommandError:
        raise
    if code != 0:
        msg = 'There was a problem running command [{m}] on host {h} over SSH, return code: {c}, and ' \
              'produced output:\n{o}'.format(h=host, c=code, m=' '.join(command), o=result['output'])
        raise CommandError(msg)
    else:
        output_text = result['output'].strip()
        log.debug('Running command [{m}] host {h} over SSH produced output: {o}'.format(
            m=command, h=host, o=output_text))
        output = {
            'output': output_text,
            'code': code
        }
    return output