def make_fileitem_peinfo_type(petype, condition='is', negate=False, preserve_case=False):
    """
    Create a node for FileItem/PEInfo/Type
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'FileItem'
    search = 'FileItem/PEInfo/Type'
    content_type = 'string'
    content = petype
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node