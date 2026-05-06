def make_fileitem_peinfo_exports_numberoffunctions(function_count, condition='is', negate=False):
    """
    Create a node for FileItem/PEInfo/Exports/NumberOfFunctions
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'FileItem'
    search = 'FileItem/PEInfo/Exports/NumberOfFunctions'
    content_type = 'int'
    content = function_count
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate)
    return ii_node