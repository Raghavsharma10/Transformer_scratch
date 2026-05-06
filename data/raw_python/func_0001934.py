def make_eventlogitem_log(log, condition='is', negate=False, preserve_case=False):
    """
    Create a node for EventLogItem/log
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'EventLogItem'
    search = 'EventLogItem/log'
    content_type = 'string'
    content = log
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node