def chmod(path, mode, recursive=False):
    """Emulates bash chmod command

    This method sets the file permissions to the specified mode.

    :param path: (str) Full path to the file or directory
    :param mode: (str) Mode to be set (e.g. 0755)
    :param recursive: (bool) Set True to make a recursive call
    :return: int exit code of the chmod command
    :raises CommandError
    """
    log = logging.getLogger(mod_logger + '.chmod')

    # Validate args
    if not isinstance(path, basestring):
        msg = 'path argument is not a string'
        log.error(msg)
        raise CommandError(msg)
    if not isinstance(mode, basestring):
        msg = 'mode argument is not a string'
        log.error(msg)
        raise CommandError(msg)

    # Ensure the item exists
    if not os.path.exists(path):
        msg = 'Item not found: {p}'.format(p=path)
        log.error(msg)
        raise CommandError(msg)

    # Create the chmod command
    command = ['chmod']
    # Make it recursive if specified
    if recursive:
        command.append('-R')
    command.append(mode)
    command.append(path)
    try:
        result = run_command(command)
    except CommandError:
        raise
    log.info('chmod command exited with code: {c}'.format(c=result['code']))
    return result['code']