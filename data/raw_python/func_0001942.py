def make_fileitem_peinfo_detectedanomalies_string(anomaly, condition='is', negate=False, preserve_case=False):
    """
    Create a node for FileItem/PEInfo/DetectedAnomalies/string
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'FileItem'
    search = 'FileItem/PEInfo/DetectedAnomalies/string'
    content_type = 'string'
    content = anomaly
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node