def make_fileitem_fileextension(extension, condition='is', negate=False, preserve_case=False):
    """
    Create a node for FileItem/FileExtension
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'FileItem'
    search = 'FileItem/FileExtension'
    content_type = 'string'
    content = extension
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node