def make_hookitem_hookingmodule(hooking_module, condition='contains', negate=False, preserve_case=False):
    """
    Create a node for HookItem/HookingModule
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'HookItem'
    search = 'HookItem/HookingModule'
    content_type = 'string'
    content = hooking_module
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node