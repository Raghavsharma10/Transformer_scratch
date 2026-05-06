def get_ADC_value(bus, addr, channel):
    """
    This method selects a channel and initiates conversion
    The ADC operates at 240 SPS (12 bits) with 1x gain
        One shot conversions are used, meaning a wait period is needed
        in order to acquire new data. This is done via a constant poll
        of the ready bit.
    Upon completion, a voltage value is returned to the caller.

    Usage - ADC_start(bus, SensorCluster.ADC_addr, channel_to_read)

    IMPORTANT NOTE:
        The ADC uses a 2.048V voltage reference

    """
    if channel == 1:
        INIT = 0b10000000
    elif channel == 2:
        INIT = 0b10100000
    elif channel == 3:
        INIT = 0b11000000
    elif channel == 4:
        INIT = 0b11100000
    bus.write_byte(addr, INIT)
    data = bus.read_i2c_block_data(addr, 0, 3)
    status = (data[2] & 0b10000000) >> 7
    while(status == 1):
        data = bus.read_i2c_block_data(addr, 0, 3)
        status = (data[2] & 0b10000000) >> 7
    sign = data[0] & 0b00001000
    val = ((data[0] & 0b0000111) << 8) | (data[1])
    if (sign == 1):
        val = (val ^ 0x3ff) + 1  # compute 2s complement for 12 bit val
    # Convert val to a ratiomerical ADC reading
    return float(val) * 2.048 / float(2047)