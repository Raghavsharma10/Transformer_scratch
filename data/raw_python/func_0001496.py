def add_invariant_checks(cls: type) -> None:
    """Decorate each of the class functions with invariant checks if not already decorated."""
    # Candidates for the decoration as list of (name, dir() value)
    init_name_func = None  # type: Optional[Tuple[str, Callable[..., None]]]
    names_funcs = []  # type: List[Tuple[str, Callable[..., None]]]
    names_properties = []  # type: List[Tuple[str, property]]

    # Filter out entries in the directory which are certainly not candidates for decoration.
    for name, value in [(name, getattr(cls, name)) for name in dir(cls)]:
        # We need to ignore __repr__ to prevent endless loops when generating error messages.
        # __getattribute__, __setattr__ and __delattr__ are too invasive and alter the state of the instance.
        # Hence we don't consider them "public".
        if name in ["__repr__", "__getattribute__", "__setattr__", "__delattr__"]:
            continue

        if name == "__init__":
            assert inspect.isfunction(value) or isinstance(value, _SLOT_WRAPPER_TYPE), \
                "Expected __init__ to be either a function or a slot wrapper, but got: {}".format(type(value))

            init_name_func = (name, value)
            continue

        if not inspect.isfunction(value) and not isinstance(value, _SLOT_WRAPPER_TYPE) and \
                not isinstance(value, property):
            continue

        # Ignore class methods
        if getattr(value, "__self__", None) is cls:
            continue

        # Ignore "protected"/"private" methods
        if name.startswith("_") and not (name.startswith("__") and name.endswith("__")):
            continue

        if inspect.isfunction(value) or isinstance(value, _SLOT_WRAPPER_TYPE):
            names_funcs.append((name, value))

        elif isinstance(value, property):
            names_properties.append((name, value))

        else:
            raise NotImplementedError("Unhandled directory entry of class {} for {}: {}".format(cls, name, value))

    if init_name_func:
        name, func = init_name_func
        wrapper = _decorate_with_invariants(func=func, is_init=True)
        setattr(cls, name, wrapper)

    for name, func in names_funcs:
        wrapper = _decorate_with_invariants(func=func, is_init=False)
        setattr(cls, name, wrapper)

    for name, prop in names_properties:
        new_prop = property(  # type: ignore
            fget=_decorate_with_invariants(func=prop.fget, is_init=False) if prop.fget else None,
            fset=_decorate_with_invariants(func=prop.fset, is_init=False) if prop.fset else None,
            fdel=_decorate_with_invariants(func=prop.fdel, is_init=False) if prop.fdel else None,
            doc=prop.__doc__)
        setattr(cls, name, new_prop)