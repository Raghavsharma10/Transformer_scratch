def make_processitem_portlist_portitem_remoteip(remote_ip, condition='is', negate=False):
    """
    Create a node for ProcessItem/PortList/PortItem/remoteIP
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'ProcessItem'
    search = 'ProcessItem/PortList/PortItem/remoteIP'
    content_type = 'IP'
    content = remote_ip
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate)
    return ii_node