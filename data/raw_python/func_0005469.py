def remove_default_gateway():
    """Removes Default Gateway configuration from /etc/sysconfig/network
    and restarts networking
    
    :return: None
    :raises: OSError
    """
    log = logging.getLogger(mod_logger + '.remove_default_gateway')

    # Ensure the network script exists
    network_script = '/etc/sysconfig/network'
    if not os.path.isfile(network_script):
        log.info('Network script not found, nothing to do: {f}'.format(f=network_script))
        return
    log.debug('Found network script: {f}'.format(f=network_script))

    # Remove settings for GATEWAY and GATEWAYDEV
    log.info('Attempting to remove any default gateway configurations...')
    for line in fileinput.input(network_script, inplace=True):
        if re.search('^GATEWAY=.*', line):
            log.info('Removing GATEWAY line: {li}'.format(li=line))
        elif re.search('^GATEWAYDEV=.*', line):
            log.info('Removing GATEWAYDEV line: {li}'.format(li=line))
        else:
            log.debug('Keeping line: {li}'.format(li=line))
            sys.stdout.write(line)

    # Restart networking for the changes to take effect
    log.info('Restarting the network service...')
    try:
        service_network_restart()
    except CommandError:
        _, ex, trace = sys.exc_info()
        raise OSError('{n}: Attempted unsuccessfully to restart networking\n{e}'.format(
            n=ex.__class__.__name__, e=str(ex)))
    else:
        log.info('Successfully restarted networking')