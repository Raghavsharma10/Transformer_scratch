def make_prefetchitem_accessedfilelist_accessedfile(accessed_file, condition='contains', negate=False,
                                                    preserve_case=False):
    """
    Create a node for PrefetchItem/AccessedFileList/AccessedFile
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'PrefetchItem'
    search = 'PrefetchItem/AccessedFileList/AccessedFile'
    content_type = 'string'
    content = accessed_file
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node