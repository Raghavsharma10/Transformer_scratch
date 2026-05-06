def import_i2c_addr(bus, opt="sensors"):
    """ import_i2c_addresses will return a list of the
            currently connected I2C devices.

        This can be used a means to automatically detect
            the number of connected sensor modules.
        Modules are between int(112) and int(119)

        By default, the method will return a list
            of sensor addresses.
    """

    i2c_list = []
    for device in range(128):
        try:
            bus.read_byte(device)
            i2c_list.append((device))
        except IOError:
            pass

    if opt == "sensors":
        sensor_list = []
        for module in range(112,120):
            try:
                indx = i2c_list.index(module)
                sensor_list.append(module)
            except ValueError:
                pass
        return sensor_list

    else:
        return i2c_list