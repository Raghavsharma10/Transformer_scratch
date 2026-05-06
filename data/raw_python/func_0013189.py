def digital_write_pullup(pin_num, value, hardware_addr=0):
    """Writes the value to the input pullup specified.

    .. note:: This function is for familiarality with users of other types of
       IO board. Consider accessing the ``gppub`` attribute of a
       PiFaceDigital object:

       >>> pfd = PiFaceDigital(hardware_addr)
       >>> hex(pfd.gppub.value)
       0xff
       >>> pfd.gppub.bits[pin_num].value = 1

    :param pin_num: The pin number to write to.
    :type pin_num: int
    :param value: The value to write.
    :type value: int
    :param hardware_addr: The board to read from (default: 0)
    :type hardware_addr: int
    """
    _get_pifacedigital(hardware_addr).gppub.bits[pin_num].value = value