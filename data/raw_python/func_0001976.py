def make_serviceitem_name(name, condition='is', negate=False, preserve_case=False):
    """
    Create a node for ServiceItem/name
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'ServiceItem'
    search = 'ServiceItem/name'
    content_type = 'string'
    content = name
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node