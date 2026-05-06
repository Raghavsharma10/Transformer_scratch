def child_watch_add(*args, **kwargs):
    """child_watch_add(priority, pid, function, *data)"""
    priority, pid, function, data = _child_watch_add_get_args(*args, **kwargs)
    return GLib.child_watch_add(priority, pid, function, *data)