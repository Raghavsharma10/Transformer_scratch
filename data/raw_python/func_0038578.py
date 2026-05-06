def inject(module_name, new_globals, *additional_modules):
    """Base method for "injecting" greened modules into an imported module.  It
    imports the module specified in *module_name*, arranging things so
    that the already-imported modules in *additional_modules* are used when
    *module_name* makes its imports.

    *new_globals* is either None or a globals dictionary that gets populated
    with the contents of the *module_name* module.  This is useful when creating
    a "green" version of some other module.

    *additional_modules* should be a collection of two-element tuples, of the
    form (<name>, <module>).  If it's not specified, a default selection of
    name/module pairs is used, which should cover all use cases but may be
    slower because there are inevitably redundant or unnecessary imports.
    """
    patched_name = '__patched_module_' + module_name
    if patched_name in sys.modules:
        # returning already-patched module so as not to destroy existing
        # references to patched modules
        return sys.modules[patched_name]

    if not additional_modules:
        # supply some defaults
        additional_modules = (_select_modules() +
                              _socket_modules() +
                              _time_modules())

    # after this we are gonna screw with sys.modules, so capture the
    # state of all the modules we're going to mess with, and lock
    saver = SysModulesSaver([name for name, m in additional_modules])
    saver.save(module_name)

    # Cover the target modules so that when you import the module it
    # sees only the patched versions
    for name, mod in additional_modules:
        sys.modules[name] = mod

    ## Remove the old module from sys.modules and reimport it while
    ## the specified modules are in place
    sys.modules.pop(module_name, None)
    try:
        module = __import__(module_name, {}, {}, module_name.split('.')[:-1])

        if new_globals is not None:
            ## Update the given globals dictionary with everything from this new module
            for name in dir(module):
                if name not in __exclude:
                    new_globals[name] = getattr(module, name)

        ## Keep a reference to the new module to prevent it from dying
        sys.modules[patched_name] = module
    finally:
        saver.restore()  # Put the original modules back

    return module