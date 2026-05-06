def make_fileitem_md5sum(md5, condition='is', negate=False):
    """
    Create a node for FileItem/Md5sum
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'FileItem'
    search = 'FileItem/Md5sum'
    content_type = 'md5'
    content = md5
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate)
    return ii_node