def mangle(data_point):
    """mangle data into expected format."""
    temp_dict = {}
    temp_dict.update(data_point)
    temp_dict['utc_datetime'] = \
        datetime.datetime.utcfromtimestamp(temp_dict['time'])
    if 'solar' in data_point:
        temp_dict['GHI (W/m^2)'] = data_point['solar']['ghi']
        temp_dict['DNI (W/m^2)'] = data_point['solar']['dni']
        temp_dict['DHI (W/m^2)'] = data_point['solar']['dhi']
        temp_dict['ETR (W/m^2)'] = data_point['solar']['etr']
        del temp_dict['solar']
    else:
        temp_dict['GHI (W/m^2)'] = 0.0
        temp_dict['DNI (W/m^2)'] = 0.0
        temp_dict['DHI (W/m^2)'] = 0.0
        temp_dict['ETR (W/m^2)'] = 0.0
    return temp_dict