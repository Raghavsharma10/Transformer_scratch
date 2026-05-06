def make_systeminfoitem_hostname(hostname, condition='contains', negate=False, preserve_case=False):
    """
    Create a node for SystemInfoItem/hostname
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'SystemInfoItem'
    search = 'SystemInfoItem/hostname'
    content_type = 'string'
    content = hostname
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node