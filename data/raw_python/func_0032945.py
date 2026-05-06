def dynamic_load(module_or_member):
    """
    Dynamically loads a class or member of a class.

    If ``module_or_member`` is something like ``"a.b.c"``, will perform ``from a.b import c``.

    If ``module_or_member`` is something like ``"a"`` will perform ``import a``

    :param module_or_member: the name of a module or member of a module to import.
    :return: the returned entity, be it a module or member of a module.
    """
    parts = module_or_member.split(".")
    if len(parts) > 1:
        name_to_import = parts[-1]
        module_to_import = ".".join(parts[:-1])
    else:
        name_to_import = None
        module_to_import = module_or_member

    module = importlib.import_module(module_to_import)
    if name_to_import:
        to_return = getattr(module, name_to_import)
        if not to_return:
            raise AttributeError("{} has no attribute {}".format(module, name_to_import))
        return to_return
    else:
        return module