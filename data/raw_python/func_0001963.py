def make_processitem_handlelist_handle_name(handle_name, condition='contains', negate=False, preserve_case=False):
    """
    Create a node for ProcessItem/HandleList/Handle/Name
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'ProcessItem'
    search = 'ProcessItem/HandleList/Handle/Name'
    content_type = 'string'
    content = handle_name
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node