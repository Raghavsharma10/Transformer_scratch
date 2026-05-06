def get_xid_devices():
    """
    Returns a list of all Xid devices connected to your computer.
    """
    devices = []
    scanner = XidScanner()
    for i in range(scanner.device_count()):
        com = scanner.device_at_index(i)
        com.open()
        device = XidDevice(com)
        devices.append(device)
    return devices