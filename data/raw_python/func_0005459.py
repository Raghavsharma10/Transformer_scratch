def mkdir_p(path):
    """Emulates 'mkdir -p' in bash

    :param path: (str) Path to create
    :return: None
    :raises CommandError
    """
    log = logging.getLogger(mod_logger + '.mkdir_p')
    if not isinstance(path, basestring):
        msg = 'path argument is not a string'
        log.error(msg)
        raise CommandError(msg)
    log.info('Attempting to create directory: %s', path)
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            msg = 'Unable to create directory: {p}'.format(p=path)
            log.error(msg)
            raise CommandError(msg)