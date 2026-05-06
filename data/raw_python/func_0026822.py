def _is_call2custom_manager(node):
    """Checks if the call is being done to a custom queryset manager."""
    called = safe_infer(node.func)
    funcdef = getattr(called, '_proxied', None)
    return _is_custom_qs_manager(funcdef)