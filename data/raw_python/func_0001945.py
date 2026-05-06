def make_fileitem_peinfo_digitalsignature_signatureverified(sig_verified, condition='is', negate=False):
    """
    Create a node for FileItem/PEInfo/DigitalSignature/SignatureVerified
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'FileItem'
    search = 'FileItem/PEInfo/DigitalSignature/SignatureVerified'
    content_type = 'bool'
    content = sig_verified
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate)
    return ii_node