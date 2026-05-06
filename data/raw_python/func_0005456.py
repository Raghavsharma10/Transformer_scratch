def get_ip_addresses():
    """Gets the ip addresses from ifconfig

    :return: (dict) of devices and aliases with the IPv4 address
    """
    log = logging.getLogger(mod_logger + '.get_ip_addresses')

    command = ['/sbin/ifconfig']
    try:
        result = run_command(command)
    except CommandError:
        raise
    ifconfig = result['output'].strip()

    # Scan the ifconfig output for IPv4 addresses
    devices = {}
    parts = ifconfig.split()
    device = None
    for part in parts:

        if device is None:
            if 'eth' in part or 'eno' in part:
                device = part
        else:
            test = part.split(':', 1)
            if len(test) == 2:
                if test[0] == 'addr':
                    ip_address = test[1]
                    log.info('Found IP address %s on device %s', ip_address,
                             device)
                    devices[device] = ip_address
                    device = None
    return devices