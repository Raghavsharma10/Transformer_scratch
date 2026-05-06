def make_serviceitem_descriptivename(descriptive_name, condition='is', negate=False, preserve_case=False):
    """
    Create a node for ServiceItem/descriptiveName
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'ServiceItem'
    search = 'ServiceItem/descriptiveName'
    content_type = 'string'
    content = descriptive_name
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node