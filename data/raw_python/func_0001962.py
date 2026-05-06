def make_prefetchitem_applicationfullpath(application_fullpath, condition='contains', negate=False,
                                          preserve_case=False):
    """
    Create a node for PrefetchItem/ApplicationFullPath
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'PrefetchItem'
    search = 'PrefetchItem/ApplicationFullPath'
    content_type = 'string'
    content = application_fullpath
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node