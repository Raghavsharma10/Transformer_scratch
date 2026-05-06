def make_processitem_sectionlist_memorysection_peinfo_exports_exportedfunctions_string(export_function, condition='is',
                                                                                       negate=False,
                                                                                       preserve_case=False):
    """
    Create a node for ProcessItem/SectionList/MemorySection/PEInfo/Exports/ExportedFunctions/string
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'ProcessItem'
    search = 'ProcessItem/SectionList/MemorySection/PEInfo/Exports/ExportedFunctions/string'
    content_type = 'string'
    content = export_function
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node