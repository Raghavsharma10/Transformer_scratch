def lazy_callable(modname, *names, **kwargs):
    """Performs lazy importing of one or more callables.

    :func:`lazy_callable` creates functions that are thin wrappers that pass
    any and all arguments straight to the target module's callables. These can
    be functions or classes. The full loading of that module is only actually
    triggered when the returned lazy function itself is called. This lazy
    import of the target module uses the same mechanism as
    :func:`lazy_module`.
    
    If, however, the target module has already been fully imported prior
    to invocation of :func:`lazy_callable`, then the target callables
    themselves are returned and no lazy imports are made.

    :func:`lazy_function` and :func:`lazy_function` are aliases of
    :func:`lazy_callable`.

    Parameters
    ----------
    modname : str
         The base module from where to import the callable(s) in *names*,
         or a full 'module_name.callable_name' string.
    names : str (optional)
         The callable name(s) to import from the module specified by *modname*.
         If left empty, *modname* is assumed to also include the callable name
         to import.
    error_strings : dict, optional
         A dictionary of strings to use when reporting loading errors (either a
         missing module, or a missing callable name in the loaded module).
         *error_string* follows the same usage as described under
         :func:`lazy_module`, with the exceptions that 1) a further key,
         'msg_callable', can be supplied to be used as the error when a module
         is successfully loaded but the target callable can't be found therein
         (defaulting to :attr:`lazy_import._MSG_CALLABLE`); 2) a key 'callable'
         is always added with the callable name being loaded.
    lazy_mod_class : type, optional
         See definition under :func:`lazy_module`.
    lazy_call_class : type, optional
         Analogously to *lazy_mod_class*, allows setting a custom class to
         handle lazy callables, other than the default :class:`LazyCallable`.

    Returns
    -------
    wrapper function or tuple of wrapper functions
        If *names* is passed, returns a tuple of wrapper functions, one for
        each element in *names*.
        If only *modname* is passed it is assumed to be a full
        'module_name.callable_name' string, in which case the wrapper for the
        imported callable is returned directly, and not in a tuple.
        
    Notes
    -----
    Unlike :func:`lazy_module`, which returns a lazy module that eventually
    mutates into the fully-functional version, :func:`lazy_callable` only
    returns thin wrappers that never change. This means that the returned
    wrapper object never truly becomes the one under the module's namespace,
    even after successful loading of the module in *modname*. This is fine for
    most practical use cases, but may break code that relies on the usage of
    the returned objects oter than calling them. One such example is the lazy
    import of a class: it's fine to use the returned wrapper to instantiate an
    object, but it can't be used, for instance, to subclass from.

    Examples
    --------
    >>> import lazy_import, sys
    >>> fn = lazy_import.lazy_callable("numpy.arange")
    >>> sys.modules['numpy']
    Lazily-loaded module numpy
    >>> fn(10)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> sys.modules['numpy']
    <module 'numpy' from '/usr/local/lib/python3.5/site-packages/numpy/__init__.py'>

    >>> import lazy_import, sys
    >>> cl = lazy_import.lazy_callable("numpy.ndarray") # a class
    >>> obj = cl([1, 2]) # This works OK (and also triggers the loading of numpy)
    >>> class MySubclass(cl): # This fails because cls is just a wrapper,
    >>>     pass              #  not an actual class.

    See Also
    --------
    :func:`lazy_module`
    :class:`LazyCallable`
    :class:`LazyModule`

    """
    if not names:
        modname, _, name = modname.rpartition(".")
    lazy_mod_class = _setdef(kwargs, 'lazy_mod_class', LazyModule)
    lazy_call_class = _setdef(kwargs, 'lazy_call_class', LazyCallable)
    error_strings = _setdef(kwargs, 'error_strings', {})
    _set_default_errornames(modname, error_strings, call=True)

    if not names:
        # We allow passing a single string as 'modname.callable_name',
        # in which case the wrapper is returned directly and not as a list.
        return _lazy_callable(modname, name, error_strings.copy(),
                                lazy_mod_class, lazy_call_class)
    return tuple(_lazy_callable(modname, cname, error_strings.copy(),
                        lazy_mod_class, lazy_call_class) for cname in names)