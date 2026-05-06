def io_add_watch(*args, **kwargs):
    """io_add_watch(channel, priority, condition, func, *user_data) -> event_source_id"""
    channel, priority, condition, func, user_data = _io_add_watch_get_args(*args, **kwargs)
    return GLib.io_add_watch(channel, priority, condition, func, *user_data)