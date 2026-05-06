def make_registryitem_valuename(valuename, condition='is', negate=False, preserve_case=False):
    """
    Create a node for RegistryItem/ValueName
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'RegistryItem'
    search = 'RegistryItem/ValueName'
    content_type = 'string'
    content = valuename
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node