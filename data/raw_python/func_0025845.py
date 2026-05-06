def weave_class(klass, aspect, methods=NORMAL_METHODS, subclasses=True, lazy=False,
                owner=None, name=None, aliases=True, bases=True, bag=BrokenBag):
    """
    Low-level weaver for classes.

    .. warning:: You should not use this directly.
    """
    assert isclass(klass), "Can't weave %r. Must be a class." % klass

    if bag.has(klass):
        return Nothing

    entanglement = Rollback()
    method_matches = make_method_matcher(methods)
    logdebug("weave_class (klass=%r, methods=%s, subclasses=%s, lazy=%s, owner=%s, name=%s, aliases=%s, bases=%s)",
             klass, methods, subclasses, lazy, owner, name, aliases, bases)

    if subclasses and hasattr(klass, '__subclasses__'):
        sub_targets = klass.__subclasses__()
        if sub_targets:
            logdebug("~ weaving subclasses: %s", sub_targets)
        for sub_class in sub_targets:
            if not issubclass(sub_class, Fabric):
                entanglement.merge(weave_class(sub_class, aspect,
                                               methods=methods, subclasses=subclasses, lazy=lazy, bag=bag))
    if lazy:
        def __init__(self, *args, **kwargs):
            super(SubClass, self).__init__(*args, **kwargs)
            for attr in dir(self):
                func = getattr(self, attr, None)
                if method_matches(attr) and attr not in wrappers and isroutine(func):
                    setattr(self, attr, _checked_apply(aspect, force_bind(func)).__get__(self, SubClass))

        wrappers = {
            '__init__': _checked_apply(aspect, __init__) if method_matches('__init__') else __init__
        }
        for attr, func in klass.__dict__.items():
            if method_matches(attr):
                if ismethoddescriptor(func):
                    wrappers[attr] = _rewrap_method(func, klass, aspect)

        logdebug(" * creating subclass with attributes %r", wrappers)
        name = name or klass.__name__
        SubClass = type(name, (klass, Fabric), wrappers)
        SubClass.__module__ = klass.__module__
        module = owner or _import_module(klass.__module__)
        entanglement.merge(patch_module(module, name, SubClass, original=klass, aliases=aliases))
    else:
        original = {}
        for attr, func in klass.__dict__.items():
            if method_matches(attr):
                if isroutine(func):
                    logdebug("@ patching attribute %r (original: %r).", attr, func)
                    setattr(klass, attr, _rewrap_method(func, klass, aspect))
                else:
                    continue
                original[attr] = func
        entanglement.merge(lambda: deque((
            setattr(klass, attr, func) for attr, func in original.items()
        ), maxlen=0))
        if bases:
            super_original = set()
            for sklass in _find_super_classes(klass):
                if sklass is not object:
                    for attr, func in sklass.__dict__.items():
                        if method_matches(attr) and attr not in original and attr not in super_original:
                            if isroutine(func):
                                logdebug("@ patching attribute %r (from superclass: %s, original: %r).",
                                         attr, sklass.__name__, func)
                                setattr(klass, attr, _rewrap_method(func, sklass, aspect))
                            else:
                                continue
                            super_original.add(attr)
            entanglement.merge(lambda: deque((
                delattr(klass, attr) for attr in super_original
            ), maxlen=0))

    return entanglement