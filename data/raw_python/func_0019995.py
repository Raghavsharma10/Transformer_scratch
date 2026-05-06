def is_mod_class(mod, cls):
    """Checks if a class in a module was declared in that module.

    Args:
        mod: the module
        cls: the class
    """
    return inspect.isclass(cls) and inspect.getmodule(cls) == mod