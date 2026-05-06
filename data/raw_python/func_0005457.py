def get_mac_address(device_index=0):
    """Returns the Mac Address given a device index

    :param device_index: (int) Device index
    :return: (str) Mac address or None
    """
    log = logging.getLogger(mod_logger + '.get_mac_address')
    command = ['ip', 'addr', 'show', 'eth{d}'.format(d=device_index)]
    log.info('Attempting to find a mac address at device index: {d}'.format(d=device_index))
    try:
        result = run_command(command)
    except CommandError:
        _, ex, trace = sys.exc_info()
        log.error('There was a problem running command, unable to determine mac address: {c}\n{e}'.format(
                c=command, e=str(ex)))
        return
    ipaddr = result['output'].split()
    get_next = False
    mac_address = None
    for part in ipaddr:
        if get_next:
            mac_address = part
            log.info('Found mac address: {m}'.format(m=mac_address))
            break
        if 'link' in part:
            get_next = True
    if not mac_address:
        log.info('mac address not found for device: {d}'.format(d=device_index))
    return mac_address