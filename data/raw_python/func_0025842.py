def weave(target, aspects, **options):
    """
    Send a message to a recipient

    Args:
        target (string, class, instance, function or builtin):
            The object to weave.
        aspects (:py:obj:`aspectlib.Aspect`, function decorator or list of):
            The aspects to apply to the object.
        subclasses (bool):
            If ``True``, subclasses of target are weaved. *Only available for classes*
        aliases (bool):
            If ``True``, aliases of target are replaced.
        lazy (bool):
            If ``True`` only target's ``__init__`` method is patched, the rest of the methods are patched after
            ``__init__`` is called. *Only available for classes*.
        methods (list or regex or string):
            Methods from target to patch. *Only available for classes*

    Returns:
        aspectlib.Rollback: An object that can rollback the patches.

    Raises:
        TypeError: If target is a unacceptable object, or the specified options are not available for that type of
            object.

    .. versionchanged:: 0.4.0

        Replaced `only_methods`, `skip_methods`, `skip_magicmethods` options with `methods`.
        Renamed `on_init` option to `lazy`.
        Added `aliases` option.
        Replaced `skip_subclasses` option with `subclasses`.
    """
    if not callable(aspects):
        if not hasattr(aspects, '__iter__'):
            raise ExpectedAdvice('%s must be an `Aspect` instance, a callable or an iterable of.' % aspects)
        for obj in aspects:
            if not callable(obj):
                raise ExpectedAdvice('%s must be an `Aspect` instance or a callable.' % obj)
    assert target, "Can't weave falsy value %r." % target
    logdebug("weave (target=%s, aspects=%s, **options=%s)", target, aspects, options)

    bag = options.setdefault('bag', ObjectBag())

    if isinstance(target, (list, tuple)):
        return Rollback([
            weave(item, aspects, **options) for item in target
        ])
    elif isinstance(target, basestring):
        parts = target.split('.')
        for part in parts:
            _check_name(part)

        if len(parts) == 1:
            return weave_module(_import_module(part), aspects, **options)

        for pos in reversed(range(1, len(parts))):
            owner, name = '.'.join(parts[:pos]), '.'.join(parts[pos:])
            try:
                owner = _import_module(owner)
            except ImportError:
                continue
            else:
                break
        else:
            raise ImportError("Could not import %r. Last try was for %s" % (target, owner))

        if '.' in name:
            path, name = name.rsplit('.', 1)
            path = deque(path.split('.'))
            while path:
                owner = getattr(owner, path.popleft())

        logdebug("@ patching %s from %s ...", name, owner)
        obj = getattr(owner, name)

        if isinstance(obj, (type, ClassType)):
            logdebug("   .. as a class %r.", obj)
            return weave_class(
                obj, aspects,
                owner=owner, name=name, **options
            )
        elif callable(obj):  # or isinstance(obj, FunctionType) ??
            logdebug("   .. as a callable %r.", obj)
            if bag.has(obj):
                return Nothing
            return patch_module_function(owner, obj, aspects, force_name=name, **options)
        else:
            return weave(obj, aspects, **options)

    name = getattr(target, '__name__', None)
    if name and getattr(__builtin__, name, None) is target:
        if bag.has(target):
            return Nothing
        return patch_module_function(__builtin__, target, aspects, **options)
    elif PY3 and ismethod(target):
        if bag.has(target):
            return Nothing
        inst = target.__self__
        name = target.__name__
        logdebug("@ patching %r (%s) as instance method.", target, name)
        func = target.__func__
        setattr(inst, name, _checked_apply(aspects, func).__get__(inst, type(inst)))
        return Rollback(lambda: delattr(inst, name))
    elif PY3 and isfunction(target):
        if bag.has(target):
            return Nothing
        owner = _import_module(target.__module__)
        path = deque(target.__qualname__.split('.')[:-1])
        while path:
            owner = getattr(owner, path.popleft())
        name = target.__name__
        logdebug("@ patching %r (%s) as a property.", target, name)
        func = owner.__dict__[name]
        return patch_module(owner, name, _checked_apply(aspects, func), func, **options)
    elif PY2 and isfunction(target):
        if bag.has(target):
            return Nothing
        return patch_module_function(_import_module(target.__module__), target, aspects, **options)
    elif PY2 and ismethod(target):
        if target.im_self:
            if bag.has(target):
                return Nothing
            inst = target.im_self
            name = target.__name__
            logdebug("@ patching %r (%s) as instance method.", target, name)
            func = target.im_func
            setattr(inst, name, _checked_apply(aspects, func).__get__(inst, type(inst)))
            return Rollback(lambda: delattr(inst, name))
        else:
            klass = target.im_class
            name = target.__name__
            return weave(klass, aspects, methods='%s$' % name, **options)
    elif isclass(target):
        return weave_class(target, aspects, **options)
    elif ismodule(target):
        return weave_module(target, aspects, **options)
    elif type(target).__module__ not in ('builtins', '__builtin__') or InstanceType and isinstance(target, InstanceType):
        return weave_instance(target, aspects, **options)
    else:
        raise UnsupportedType("Can't weave object %s of type %s" % (target, type(target)))