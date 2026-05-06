def coerce_to_synchronous(func):
    '''
    Given a function that might be async, wrap it in an explicit loop so it can
    be run in a synchronous context.
    '''
    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            try:
                loop.run_until_complete(func(*args, **kwargs))
            finally:
                loop.close()
        return sync_wrapper
    return func