def list_classes(mod_name):
    """Lists all classes declared in a module.

    Args:
        mod_name: the module name
    Returns:
        A list of functions declared in that module.
    """
    mod = sys.modules[mod_name]
    return [cls.__name__ for cls in mod.__dict__.values()
            if is_mod_class(mod, cls)]