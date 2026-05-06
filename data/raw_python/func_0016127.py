def lazy_module(modname, error_strings=None, lazy_mod_class=LazyModule,
                  level='leaf'):
    """Function allowing lazy importing of a module into the namespace.

    A lazy module object is created, registered in `sys.modules`, and
    returned. This is a hollow module; actual loading, and `ImportErrors` if
    not found, are delayed until an attempt is made to access attributes of the
    lazy module.

    A handy application is to use :func:`lazy_module` early in your own code
    (say, in `__init__.py`) to register all modulenames you want to be lazy.
    Because of registration in `sys.modules` later invocations of
    `import modulename` will also return the lazy object. This means that after
    initial registration the rest of your code can use regular pyhon import
    statements and retain the lazyness of the modules.

    Parameters
    ----------
    modname : str
         The module to import.
    error_strings : dict, optional
         A dictionary of strings to use when module-loading fails. Key 'msg'
         sets the message to use (defaults to :attr:`lazy_import._MSG`). The
         message is formatted using the remaining dictionary keys. The default
         message informs the user of which module is missing (key 'module'),
         what code loaded the module as lazy (key 'caller'), and which package
         should be installed to solve the dependency (key 'install_name').
         None of the keys is mandatory and all are given smart names by default.
    lazy_mod_class: type, optional
         Which class to use when instantiating the lazy module, to allow
         deep customization. The default is :class:`LazyModule` and custom
         alternatives **must** be a subclass thereof.
    level : str, optional
         Which submodule reference to return. Either a reference to the 'leaf'
         module (the default) or to the 'base' module. This is useful if you'll
         be using the module functionality in the same place you're calling
         :func:`lazy_module` from, since then you don't need to run `import`
         again. Setting *level* does not affect which names/modules get
         registered in `sys.modules`.
         For *level* set to 'base' and *modulename* 'aaa.bbb.ccc'::

            aaa = lazy_import.lazy_module("aaa.bbb.ccc", level='base')
            # 'aaa' becomes defined in the current namespace, with
            #  (sub)attributes 'aaa.bbb' and 'aaa.bbb.ccc'.
            # It's the lazy equivalent to:
            import aaa.bbb.ccc

        For *level* set to 'leaf'::

            ccc = lazy_import.lazy_module("aaa.bbb.ccc", level='leaf')
            # Only 'ccc' becomes set in the current namespace.
            # Lazy equivalent to:
            from aaa.bbb import ccc

    Returns
    -------
    module
        The module specified by *modname*, or its base, depending on *level*.
        The module isn't immediately imported. Instead, an instance of
        *lazy_mod_class* is returned. Upon access to any of its attributes, the
        module is finally loaded.

    Examples
    --------
    >>> import lazy_import, sys
    >>> np = lazy_import.lazy_module("numpy")
    >>> np
    Lazily-loaded module numpy
    >>> np is sys.modules['numpy']
    True
    >>> np.pi # This causes the full loading of the module ...
    3.141592653589793
    >>> np # ... and the module is changed in place. 
    <module 'numpy' from '/usr/local/lib/python/site-packages/numpy/__init__.py'>

    >>> import lazy_import, sys
    >>> # The following succeeds even when asking for a module that's not available
    >>> missing = lazy_import.lazy_module("missing_module")
    >>> missing
    Lazily-loaded module missing_module
    >>> missing is sys.modules['missing_module']
    True
    >>> missing.some_attr # This causes the full loading of the module, which now fails.
    ImportError: __main__ attempted to use a functionality that requires module missing_module, but it couldn't be loaded. Please install missing_module and retry.

    See Also
    --------
    :func:`lazy_callable`
    :class:`LazyModule`

    """
    if error_strings is None:
        error_strings = {}
    _set_default_errornames(modname, error_strings)

    mod = _lazy_module(modname, error_strings, lazy_mod_class)
    if level == 'base':
        return sys.modules[module_basename(modname)]
    elif level == 'leaf':
        return mod
    else:
        raise ValueError("Parameter 'level' must be one of ('base', 'leaf')")