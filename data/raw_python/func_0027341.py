def get_device_address(device):
    """ find the local ip address on the given device """
    if device is None:
        return None
    command = ['ip', 'route', 'list', 'dev', device]
    ip_routes = subprocess.check_output(command).strip()
    for line in ip_routes.split('\n'):
        seen = ''
        for a in line.split():
            if seen == 'src':
                return a
            seen = a
    return None