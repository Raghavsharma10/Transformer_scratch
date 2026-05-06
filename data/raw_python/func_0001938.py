def make_fileitem_filename(filename, condition='is', negate=False, preserve_case=False):
    """
    Create a node for FileItem/FileName
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'FileItem'
    search = 'FileItem/FileName'
    content_type = 'string'
    content = filename
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node