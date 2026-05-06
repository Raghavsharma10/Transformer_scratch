def make_serviceitem_description(description, condition='contains', negate=False, preserve_case=False):
    """
    Create a node for ServiceItem/description
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'ServiceItem'
    search = 'ServiceItem/description'
    content_type = 'string'
    content = description
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node