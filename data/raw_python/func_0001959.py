def make_portitem_remoteport(remote_port, condition='is', negate=False):
    """
    Create a node for PortItem/remotePort
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'PortItem'
    search = 'PortItem/remotePort'
    content_type = 'int'
    content = remote_port
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate)
    return ii_node