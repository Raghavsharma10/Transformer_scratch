def is_mod_function(mod, fun):
    """Checks if a function in a module was declared in that module.

    http://stackoverflow.com/a/1107150/3004221

    Args:
        mod: the module
        fun: the function
    """
    return inspect.isfunction(fun) and inspect.getmodule(fun) == mod