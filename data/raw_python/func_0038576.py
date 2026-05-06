def original(modname):
    """ This returns an unpatched version of a module."""
    # note that it's not necessary to temporarily install unpatched
    # versions of all patchable modules during the import of the
    # module; this is because none of them import each other, except
    # for threading which imports thread
    original_name = '__original_module_' + modname
    if original_name in sys.modules:
        return sys.modules.get(original_name)

    # re-import the "pure" module and store it in the global _originals
    # dict; be sure to restore whatever module had that name already
    saver = SysModulesSaver((modname,))
    sys.modules.pop(modname, None)
    try:
        real_mod = __import__(modname, {}, {}, modname.split('.')[:-1])
        # save a reference to the unpatched module so it doesn't get lost
        sys.modules[original_name] = real_mod
    finally:
        saver.restore()

    return sys.modules[original_name]