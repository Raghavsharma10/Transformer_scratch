def qualified_name(cls):
    """Full name of a class, including the module. Like qualified_class_name, but when you already have a class """
    module = cls.__module__
    if module is None or module == str.__class__.__module__:
        return cls.__name__
    return module + '.' + cls.__name__