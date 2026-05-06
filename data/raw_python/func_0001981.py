def make_serviceitem_servicedllmd5sum(servicedll_md5, condition='is', negate=False):
    """
    Create a node for ServiceItem/serviceDLLmd5sum
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'ServiceItem'
    search = 'ServiceItem/serviceDLLmd5sum'
    content_type = 'md5'
    content = servicedll_md5
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate)
    return ii_node