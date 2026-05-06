def make_fileitem_username(file_owner, condition='is', negate=False, preserve_case=False):
    """
    Create a node for FileItem/Username
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'FileItem'
    search = 'FileItem/Username'
    content_type = 'string'
    content = file_owner
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node