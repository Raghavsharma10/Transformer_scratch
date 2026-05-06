def start(io_loop=None, check_time=2):

    """Begins watching source files for changes.

    .. versionchanged:: 4.1
       The ``io_loop`` argument is deprecated.
    """
    io_loop = io_loop or asyncio.get_event_loop()
    if io_loop in _io_loops:
        return
    _io_loops[io_loop] = True
    if len(_io_loops) > 1:
        logger.warning("aiohttp_autoreload started more than once in the same process")
    # if _has_execv:
    #     add_reload_hook(functools.partial(io_loop.close, all_fds=True))
    modify_times = {}
    callback = functools.partial(_reload_on_update, modify_times)
    logger.debug("Starting periodic checks for code changes")
    call_periodic(check_time, callback, loop=io_loop)