def IO_expander_output(bus, addr, bank, mask):
    """
    Method for controlling the GPIO expander via I2C
        which accepts a bank - A(0) or B(1) and a mask
        to push to the pins of the expander.

    The method also assumes the the expander is operating
        in sequential mode. If this mode is not used,
        the register addresses will need to be changed.

    Usage:
    GPIO_out(bus, GPIO_addr, 0, 0b00011111)
        This call would turn on A0 through A4. 

    """
    IODIR_map = [0x00, 0x01]
    output_map = [0x14, 0x15]

    if (bank != 0) and (bank != 1):
        print()
        raise InvalidIOUsage("An invalid IO bank has been selected")


    IO_direction = IODIR_map[bank]
    output_reg = output_map[bank]

    current_status = bus.read_byte_data(addr, output_reg)
    if current_status == mask:
        # This means nothing needs to happen
        print("Current control status matches requested controls. " +
              "No action is required.")
        return True

    bus.write_byte_data(addr, IO_direction, 0)
    bus.write_byte_data(addr, output_reg, mask)