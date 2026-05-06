def digital_write(pin_num, value, hardware_addr=0):
    """Writes the value to the input pin specified.

    .. note:: This function is for familiarality with users of other types of
       IO board. Consider accessing the ``output_pins`` attribute of a
       PiFaceDigital object:

       >>> pfd = PiFaceDigital(hardware_addr)
       >>> pfd.output_pins[pin_num].value = 1

    :param pin_num: The pin number to write to.
    :type pin_num: int
    :param value: The value to write.
    :type value: int
    :param hardware_addr: The board to read from (default: 0)
    :type hardware_addr: int
    """
    _get_pifacedigital(hardware_addr).output_pins[pin_num].value = value