def make_processitem_path(path, condition='contains', negate=False, preserve_case=False):
    """
    Create a node for ProcessItem/path
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'ProcessItem'
    search = 'ProcessItem/path'
    content_type = 'string'
    content = path
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node