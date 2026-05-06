def make_eventlogitem_message(message, condition='contains', negate=False, preserve_case=False):
    """
    Create a node for EventLogItem/message
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'EventLogItem'
    search = 'EventLogItem/message'
    content_type = 'string'
    content = message
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node