def make_serviceitem_pathmd5sum(path_md5, condition='is', negate=False):
    """
    Create a node for ServiceItem/pathmd5sum
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'ServiceItem'
    search = 'ServiceItem/pathmd5sum'
    content_type = 'md5'
    content = path_md5
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate)
    return ii_node