def hook(name=None, priority=-1):
    """
    Decorator
    """

    def _hook(hook_func):
        return register_hook(name, hook_func=hook_func, priority=priority)

    return _hook