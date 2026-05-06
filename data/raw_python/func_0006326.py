def interpret_data_from_tektronix(preamble, data):
    ''' Interprets raw data from Tektronix
    returns: lists of x, y values in seconds/volt'''
    # Y mode ("WFMPRE:PT_FMT"):
    # Xn = XZEro + XINcr (n - PT_Off)
    # Yn = YZEro + YMUlt (yn - YOFf)
    voltage = np.array(data, dtype=np.float)
    meta_data = preamble.split(',')[5].split(';')
    time_unit = meta_data[3][1:-1]
    XZEro = float(meta_data[5])
    XINcr = float(meta_data[4])
    PT_Off = float(meta_data[6])
    voltage_unit = meta_data[7][1:-1]
    YZEro = float(meta_data[10])
    YMUlt = float(meta_data[8])
    YOFf = float(meta_data[9])
    time = XZEro + XINcr * (np.arange(0, voltage.size) - PT_Off)
    voltage = YZEro + YMUlt * (voltage - YOFf)
    return time, voltage, time_unit, voltage_unit