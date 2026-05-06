def make_driveritem_drivername(driver_name, condition='contains', negate=False, preserve_case=False):
    """
    Create a node for DriverItem/DriverName
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'DriverItem'
    search = 'DriverItem/DriverName'
    content_type = 'string'
    content = driver_name
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node