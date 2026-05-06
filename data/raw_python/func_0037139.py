def get_IO_reg(bus, addr, bank):
    """
    Method retrieves the register corresponding to respective bank (0 or 1)
    """
    output_map = [0x14, 0x15]
    if (bank != 0) and (bank != 1):
        print()
        raise InvalidIOUsage("An invalid IO bank has been selected")

    output_reg = output_map[bank]
    current_status = bus.read_byte_data(addr, output_reg)
    return current_status