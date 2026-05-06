def list_functions(mod_name):
    """Lists all functions declared in a module.

    http://stackoverflow.com/a/1107150/3004221

    Args:
        mod_name: the module name
    Returns:
        A list of functions declared in that module.
    """
    mod = sys.modules[mod_name]
    return [func.__name__ for func in mod.__dict__.values()
            if is_mod_function(mod, func)]