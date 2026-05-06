def make_registryitem_keypath(keypath, condition='contains', negate=False, preserve_case=False):
    """
    Create a node for RegistryItem/KeyPath
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'RegistryItem'
    search = 'RegistryItem/KeyPath'
    content_type = 'string'
    content = keypath
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node