def is_module(obj):
    """
    Checking and setting type to MODULE
    Args:
        obj: ModuleType / class
        Note: An instance will be treated as a Class
    Return:
        Boolean
    """
    return True if obj and isinstance(obj, ModuleType) or inspect.isclass(obj) else False