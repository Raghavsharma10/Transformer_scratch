def get_lux_count(lux_byte):
    """ Method to convert data from the TSL2550D lux sensor
    into more easily usable ADC count values.

    """
    LUX_VALID_MASK = 0b10000000
    LUX_CHORD_MASK = 0b01110000
    LUX_STEP_MASK = 0b00001111
    valid = lux_byte & LUX_VALID_MASK
    if valid != 0:
        step_num = (lux_byte & LUX_STEP_MASK)
        # Shift to normalize value
        chord_num = (lux_byte & LUX_CHORD_MASK) >> 4
        step_val = 2**chord_num
        chord_val = int(16.5 * (step_val - 1))
        count = chord_val + step_val * step_num
        return count
    else:
        raise SensorError("Invalid lux sensor data.")