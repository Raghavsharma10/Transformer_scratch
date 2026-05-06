def make_serviceitem_servicedll(servicedll, condition='contains', negate=False, preserve_case=False):
    """
    Create a node for ServiceItem/serviceDLL
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'ServiceItem'
    search = 'ServiceItem/serviceDLL'
    content_type = 'string'
    content = servicedll
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node