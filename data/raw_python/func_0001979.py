def make_serviceitem_servicedllsignatureexists(dll_sig_exists, condition='is', negate=False):
    """
    Create a node for ServiceItem/serviceDLLSignatureExists
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'ServiceItem'
    search = 'ServiceItem/serviceDLLSignatureExists'
    content_type = 'bool'
    content = dll_sig_exists
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate)
    return ii_node