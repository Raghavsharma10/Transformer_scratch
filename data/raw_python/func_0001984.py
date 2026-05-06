def make_fileitem_peinfo_versioninfoitem(key, value, condition='is', negate=False, preserve_case=False):
    """
    Create a node for FileItem/PEInfo/VersionInfoList/VersionInfoItem/ + key name
    
    No validation of the key is performed.
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'FileItem'
    search = 'FileItem/PEInfo/VersionInfoList/VersionInfoItem/' + key  # XXX: No validation of key done.
    content_type = 'string'
    content = value
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node