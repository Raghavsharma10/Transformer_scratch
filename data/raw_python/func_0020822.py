def open(port=None, serial_number=None):
    """Open an aardvark device and return an :class:`Aardvark` object. If the
    device cannot be opened an :class:`IOError` is raised.

    The `port` can be retrieved by :func:`find_devices`. Usually, the first
    device is 0, the second 1, etc.

    If you are using only one device, you can therefore omit the parameter
    in which case 0 is used.

    Another method to open a device is to use the serial number. You can either
    find the number on the device itself or in the in the corresponding USB
    property. The serial number is a string which looks like `NNNN-MMMMMMM`.

    Raises an :class:`IOError` if the port (or serial number) does not exist,
    is already connected or an incompatible device is found.

    .. note::

       There is a small chance that this function raises an :class:`IOError`
       although the correct device is available and not opened. The
       open-by-serial-number method works by scanning the devices. But as
       explained in :func:`find_devices`, the returned information may be
       outdated. Therefore, :func:`open` checks the serial number once the
       device is opened and if it is not the expected one, raises
       :class:`IOError`. No retry mechanism is implemented.

       As long as nobody comes along with a better idea, this failure case is
       up to the user.
    """
    if port is None and serial_number is None:
        dev = Aardvark()
    elif serial_number is not None:
        for d in find_devices():
            if d['serial_number'] == serial_number:
                break
        else:
            _raise_error_if_negative(ERR_UNABLE_TO_OPEN)

        dev = Aardvark(d['port'])

        # make sure we opened the correct device
        if dev.unique_id_str() != serial_number:
            dev.close()
            _raise_error_if_negative(ERR_UNABLE_TO_OPEN)
    else:
        dev = Aardvark(port)

    return dev