def make_fileitem_filepath(filepath, condition='contains', negate=False, preserve_case=False):
    """
    Create a node for FileItem/FilePath
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'FileItem'
    search = 'FileItem/FilePath'
    content_type = 'string'
    content = filepath
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node