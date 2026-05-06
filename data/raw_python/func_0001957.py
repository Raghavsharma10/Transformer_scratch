def make_hookitem_hookedfunction(hooked_function, condition='is', negate=False, preserve_case=False):
    """
    Create a node for HookItem/HookedFunction
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'HookItem'
    search = 'HookItem/HookedFunction'
    content_type = 'string'
    content = hooked_function
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node