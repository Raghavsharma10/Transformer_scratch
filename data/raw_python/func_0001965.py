def make_processitem_sectionlist_memorysection_name(section_name, condition='contains', negate=False,
                                                    preserve_case=False):
    """
    Create a node for ProcessItem/SectionList/MemorySection/Name
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'ProcessItem'
    search = 'ProcessItem/SectionList/MemorySection/Name'
    content_type = 'string'
    content = section_name
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node