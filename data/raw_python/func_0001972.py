def make_registryitem_text(text, condition='contains', negate=False, preserve_case=False):
    """
    Create a node for RegistryItem/Text
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'RegistryItem'
    search = 'RegistryItem/Text'
    content_type = 'string'
    content = text
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node