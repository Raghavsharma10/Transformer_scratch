def get_xid_device(device_number):
    """
    returns device at a given index.

    Raises ValueError if the device at the passed in index doesn't
    exist.
    """
    scanner = XidScanner()
    com = scanner.device_at_index(device_number)
    com.open()
    return XidDevice(com)