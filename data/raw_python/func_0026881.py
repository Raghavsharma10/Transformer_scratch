def calculate_manual_reading(basic_data: BasicMeterData) -> Reading:
    """ Calculate the interval between two manual readings """
    t_start = basic_data.previous_register_read_datetime
    t_end = basic_data.current_register_read_datetime
    read_start = basic_data.previous_register_read
    read_end = basic_data.current_register_read
    value = basic_data.quantity

    uom = basic_data.uom
    quality_method = basic_data.current_quality_method

    return Reading(t_start, t_end, value, uom, quality_method, "", "",
                      read_start, read_end)