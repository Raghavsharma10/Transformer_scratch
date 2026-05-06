def global_import(mod_name):
    """
    This function search sys.path[1:], return specified module.
    sys.path[1:] means the directories other than current directory('./').

    'mod_name' as an argument is string type object.
    """
    mod_tuple = imp.find_module(mod_name, sys.path[1:])
    mod = imp.load_module(mod_name, mod_tuple[0], mod_tuple[1], mod_tuple[2])

    return mod