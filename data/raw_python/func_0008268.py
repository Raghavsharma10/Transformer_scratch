def deep_getattr(clsdict: Dict[str, Any],
                 bases: Tuple[Type[object], ...],
                 name: str,
                 default: Any = _missing) -> Any:
    """
    Acts just like ``getattr`` would on a constructed class object, except this operates
    on the pre-construction class dictionary and base classes. In other words, first we
    look for the attribute in the class dictionary, and then we search all the base
    classes (in method resolution order), finally returning the default value if the
    attribute was not found in any of the class dictionary or base classes (or it raises
    ``AttributeError`` if no default was given).
    """
    value = clsdict.get(name, _missing)
    if value != _missing:
        return value
    for base in bases:
        value = getattr(base, name, _missing)
        if value != _missing:
            return value
    if default != _missing:
        return default
    raise AttributeError(name)