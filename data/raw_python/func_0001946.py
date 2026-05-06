def make_fileitem_peinfo_exports_dllname(dll_name, condition='is', negate=False, preserve_case=False):
    """
    Create a node for FileItem/PEInfo/Exports/DllName
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'FileItem'
    search = 'FileItem/PEInfo/Exports/DllName'
    content_type = 'string'
    content = dll_name
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node