def make_serviceitem_servicedllsignatureverified(dll_sig_verified, condition='is', negate=False):
    """
    Create a node for ServiceItem/serviceDLLSignatureVerified
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'ServiceItem'
    search = 'ServiceItem/serviceDLLSignatureVerified'
    content_type = 'bool'
    content = dll_sig_verified
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate)
    return ii_node