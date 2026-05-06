def make_processitem_arguments(arguments, condition='contains', negate=False, preserve_case=False):
    """
    Create a node for ProcessItem/arguments
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'ProcessItem'
    search = 'ProcessItem/arguments'
    content_type = 'string'
    content = arguments
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node