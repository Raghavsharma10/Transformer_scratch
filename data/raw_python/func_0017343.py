def _make_hook_dict(fun):
    """Ensure the given function has a xworkflows_hook attribute.

    That attribute has the following structure:
    >>> {
    ...     'before': [('state', <TransitionHook>), ...],
    ... }
    """
    if not hasattr(fun, 'xworkflows_hook'):
        fun.xworkflows_hook = {
            HOOK_BEFORE: [],
            HOOK_AFTER: [],
            HOOK_CHECK: [],
            HOOK_ON_ENTER: [],
            HOOK_ON_LEAVE: [],
        }
    return fun.xworkflows_hook