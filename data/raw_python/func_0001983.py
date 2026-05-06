def make_systemrestoreitem_originalfilename(original_filename, condition='contains', negate=False, preserve_case=False):
    """
    Create a node for SystemRestoreItem/OriginalFileName
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'SystemRestoreItem'
    search = 'SystemRestoreItem/OriginalFileName'
    content_type = 'string'
    content = original_filename
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node