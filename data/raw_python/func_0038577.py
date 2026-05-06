def patch(**on):
    """Globally patches certain system modules to be 'cooperaive'.

    The keyword arguments afford some control over which modules are patched.
    If no keyword arguments are supplied, all possible modules are patched.
    If keywords are set to True, only the specified modules are patched.  E.g.,
    ``monkey_patch(socket=True, select=True)`` patches only the select and
    socket modules.  Most arguments patch the single module of the same name
    (os, time, select).  The exception is socket, which also patches the ssl
    module if present.

    It's safe to call monkey_patch multiple times.
    """
    accepted_args = set(('select', 'socket', 'time'))
    default_on = on.pop("all", None)
    for k in on.keys():
        if k not in accepted_args:
            raise TypeError("patch() got an unexpected keyword argument %r" % k)
    if default_on is None:
        default_on = not (True in list(on.values()))
    for modname in accepted_args:
        on.setdefault(modname, default_on)

    modules_to_patch = []
    if on['select'] and not already_patched.get('select'):
        modules_to_patch += _select_modules()
        already_patched['select'] = True
    if on['socket'] and not already_patched.get('socket'):
        modules_to_patch += _socket_modules()
        already_patched['socket'] = True
    if on['time'] and not already_patched.get('time'):
        modules_to_patch += _time_modules()
        already_patched['time'] = True

    imp.acquire_lock()
    try:
        for name, mod in modules_to_patch:
            orig_mod = sys.modules.get(name)
            if orig_mod is None:
                orig_mod = __import__(name)
            for attr_name in mod.__patched__:
                patched_attr = getattr(mod, attr_name, None)
                if patched_attr is not None:
                    setattr(orig_mod, attr_name, patched_attr)
    finally:
        imp.release_lock()