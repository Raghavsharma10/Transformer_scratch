def make_driveritem_deviceitem_devicename(device_name, condition='is', negate=False, preserve_case=False):
    """
    Create a node for DriverItem/DeviceItem/DeviceName
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'DriverItem'
    search = 'DriverItem/DeviceItem/DeviceName'
    content_type = 'string'
    content = device_name
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node