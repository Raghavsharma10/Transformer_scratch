def find_devices():
    """Return a list of dictionaries. Each dictionary represents one device.

    The dictionary contains the following keys: port, unique_id and in_use.
    `port` can be used with :func:`open`. `serial_number` is the serial number
    of the device (and can also be used with :func:`open`) and `in_use`
    indicates whether the device was opened before and can currently not be
    opened.

    .. note::

       There is no guarantee, that the returned information is still valid
       when you open the device. Esp. if you open a device by the port, the
       unique_id may change because you've just opened another device. Eg. it
       may be disconnected from the machine after you call :func:`find_devices`
       but before you call :func:`open`.

       To open a device by its serial number, you should use the :func:`open`
       with the `serial_number` parameter.
    """

    # first fetch the number of attached devices, so we can create a buffer
    # with the exact amount of entries. api expects array of u16
    num_devices = api.py_aa_find_devices(0, array.array('H'))
    _raise_error_if_negative(num_devices)

    # return an empty list if no device is connected
    if num_devices == 0:
        return list()

    ports = array.array('H', (0,) * num_devices)
    unique_ids = array.array('I', (0,) * num_devices)
    num_devices = api.py_aa_find_devices_ext(len(ports), len(unique_ids),
            ports, unique_ids)
    _raise_error_if_negative(num_devices)
    if num_devices == 0:
        return list()

    del ports[num_devices:]
    del unique_ids[num_devices:]

    devices = list()
    for port, uid in zip(ports, unique_ids):
        in_use = bool(port & PORT_NOT_FREE)
        dev = dict(
                port=port & ~PORT_NOT_FREE,
                serial_number=_unique_id_str(uid),
                in_use=in_use)
        devices.append(dev)

    return devices