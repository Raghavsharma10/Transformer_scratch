def check_remote_host_marker_file(host, file_path):
    """Queries a remote host over SSH to check for existence
    of a marker file

    :param host: (str) host to query
    :param file_path: (str) path to the marker file
    :return: (bool) True if the marker file exists
    :raises: TypeError, CommandError
    """
    log = logging.getLogger(mod_logger + '.check_remote_host_marker_file')
    if not isinstance(host, basestring):
        msg = 'host argument must be a string'
        log.error(msg)
        raise TypeError(msg)
    if not isinstance(file_path, basestring):
        msg = 'file_path argument must be a string'
        log.error(msg)
        raise TypeError(msg)
    log.debug('Checking host {h} for marker file: {f}...'.format(h=host, f=file_path))
    command = ['ssh', '{h}'.format(h=host), 'if [ -f {f} ] ; then exit 0 ; else exit 1 ; fi'.format(f=file_path)]
    try:
        result = run_command(command, timeout_sec=5.0)
        code = result['code']
        output = result['output']
    except CommandError:
        raise
    if code == 0:
        log.debug('Marker file <{f}> was found on host {h}'.format(f=file_path, h=host))
        return True
    elif code == 1 and output == '':
        log.debug('Marker file <{f}> was not found on host {h}'.format(f=file_path, h=host))
        return False
    else:
        msg = 'There was a problem checking the remote host {h} over SSH for marker file {f}, ' \
              'command returned code {c} and produced output: {o}'.format(
                h=host, f=file_path, c=code, o=output)
        log.debug(msg)
        raise CommandError(msg)