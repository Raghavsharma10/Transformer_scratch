def make_fileitem_sizeinbytes(filesize, condition='is', negate=False):
    """
    Create a node for FileItem/SizeInBytes
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'FileItem'
    search = 'FileItem/SizeInBytes'
    content_type = 'int'
    content = filesize
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate)
    return ii_node