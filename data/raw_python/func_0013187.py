def deinit(bus=DEFAULT_SPI_BUS,
           chip_select=DEFAULT_SPI_CHIP_SELECT):
    """Stops interrupts on all boards. Only required when using
    :func:`digital_read` and :func:`digital_write`.

    :param bus: SPI bus /dev/spidev<bus>.<chipselect> (default: {bus})
    :type bus: int
    :param chip_select: SPI chip select /dev/spidev<bus>.<chipselect>
        (default: {chip})
    :type chip_select: int
    """
    global _pifacedigitals
    for pfd in _pifacedigitals:
        try:
            pfd.deinit_board()
        except AttributeError:
            pass