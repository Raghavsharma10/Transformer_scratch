def get_default_net_device():
    """ Find the device where the default route is. """
    with open('/proc/net/route') as fh:
        for line in fh:
            iface, dest, _ = line.split(None, 2)
            if dest == '00000000':
                return iface
    return None