def make_eventlogitem_eid(eid, condition='is', negate=False):
    """
    Create a node for EventLogItem/EID
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'EventLogItem'
    search = 'EventLogItem/EID'
    content_type = 'int'
    content = eid
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate)
    return ii_node