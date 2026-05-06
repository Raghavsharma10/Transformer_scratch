def make_fileitem_streamlist_stream_name(stream_name, condition='is', negate=False, preserve_case=False):
    """
    Create a node for FileItem/StreamList/Stream/Name
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'FileItem'
    search = 'FileItem/StreamList/Stream/Name'
    content_type = 'string'
    content = stream_name
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node