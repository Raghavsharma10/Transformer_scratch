def make_fileitem_peinfo_digitalsignature_signatureexists(sig_exists, condition='is', negate=False):
    """
    Create a node for FileItem/PEInfo/DigitalSignature/SignatureExists
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'FileItem'
    search = 'FileItem/PEInfo/DigitalSignature/SignatureExists'
    content_type = 'bool'
    content = sig_exists
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate)
    return ii_node