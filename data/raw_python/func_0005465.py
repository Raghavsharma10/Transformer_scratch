def set_remote_host_environment_variable(host, variable_name, variable_value, env_file='/etc/bashrc'):
    """Sets an environment variable on the remote host in the
    specified environment file

    :param host: (str) host to set environment variable on
    :param variable_name: (str) name of the variable
    :param variable_value: (str) value of the variable
    :param env_file: (str) full path to the environment file to set
    :return: None
    :raises: TypeError, CommandError
    """
    log = logging.getLogger(mod_logger + '.set_remote_host_environment_variable')
    if not isinstance(host, basestring):
        msg = 'host argument must be a string'
        log.error(msg)
        raise TypeError(msg)
    if not isinstance(variable_name, basestring):
        msg = 'variable_name argument must be a string'
        log.error(msg)
        raise TypeError(msg)
    if not isinstance(variable_value, basestring):
        msg = 'variable_value argument must be a string'
        log.error(msg)
        raise TypeError(msg)
    if not isinstance(env_file, basestring):
        msg = 'env_file argument must be a string'
        log.error(msg)
        raise TypeError(msg)
    log.info('Creating the environment file if it does not exist...')
    command = ['ssh', host, 'touch {f}'.format(f=env_file)]
    try:
        result = run_command(command, timeout_sec=5.0)
        code = result['code']
        output = result['output']
    except CommandError:
        raise
    if code != 0:
        msg = 'There was a problem creating environment file {f} on remote host {h} over SSH, ' \
              'exit code {c} and output:\n{o}'.format(h=host, c=code, f=env_file, o=output)
        log.error(msg)
        raise CommandError(msg)

    log.info('Creating ensuring the environment file is executable...')
    command = ['ssh', host, 'chmod +x {f}'.format(f=env_file)]
    try:
        result = run_command(command, timeout_sec=5.0)
        code = result['code']
        output = result['output']
    except CommandError:
        raise
    if code != 0:
        msg = 'There was a problem setting permissions on environment file {f} on remote host {h} over SSH, ' \
              'exit code {c} and output:\n{o}'.format(h=host, c=code, f=env_file, o=output)
        log.error(msg)
        raise CommandError(msg)

    log.info('Adding environment variable {v} with value {n} to file {f}...'.format(
            v=variable_name, n=variable_value, f=env_file))
    command = ['ssh', host, 'echo "export {v}=\\"{n}\\"" >> {f}'.format(f=env_file, v=variable_name, n=variable_value)]
    try:
        result = run_command(command, timeout_sec=5.0)
        code = result['code']
        output = result['output']
    except CommandError:
        raise
    if code != 0:
        msg = 'There was a problem adding variable {v} to environment file {f} on remote host {h} over SSH, ' \
              'exit code {c} and output:\n{o}'.format(h=host, c=code, f=env_file, o=output, v=variable_name)
        log.error(msg)
        raise CommandError(msg)
    else:
        log.info('Environment variable {v} set to {n} on host {h}'.format(v=variable_name, n=variable_value, h=host))