def rpm_install(install_dir):
    """This method installs all RPM files in a specific dir

    :param install_dir: (str) Full path to the directory
    :return int exit code form the rpm command
    :raises CommandError
    """
    log = logging.getLogger(mod_logger + '.rpm_install')

    # Type checks on the args
    if not isinstance(install_dir, basestring):
        msg = 'install_dir argument must be a string'
        log.error(msg)
        raise CommandError(msg)

    # Ensure the install_dir directory exists
    if not os.path.isdir(install_dir):
        msg = 'Directory not found: {f}'.format(f=install_dir)
        log.error(msg)
        raise CommandError(msg)

    # Create the command
    command = ['rpm', '-iv', '--force', '{d}/*.rpm'.format(d=install_dir)]

    # Run the rpm command
    try:
        result = run_command(command)
    except CommandError:
        raise
    log.info('RPM completed and exit with code: {c}'.format(
        c=result['code']))
    return result['code']