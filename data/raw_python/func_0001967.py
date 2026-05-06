def make_processitem_stringlist_string(string, condition='contains', negate=False, preserve_case=False):
    """
    Create a node for ProcessItem/StringList/string
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'ProcessItem'
    search = 'ProcessItem/StringList/string'
    content_type = 'string'
    content = string
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node