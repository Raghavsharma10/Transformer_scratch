def _launch_forever_coro(coro, args, kwargs, loop):
    '''
    This helper function launches an async main function that was tagged with
    forever=True. There are two possibilities:

    - The function is a normal function, which handles initializing the event
      loop, which is then run forever
    - The function is a coroutine, which needs to be scheduled in the event
      loop, which is then run forever
      - There is also the possibility that the function is a normal function
        wrapping a coroutine function

    The function is therefore called unconditionally and scheduled in the event
    loop if the return value is a coroutine object.

    The reason this is a separate function is to make absolutely sure that all
    the objects created are garbage collected after all is said and done; we
    do this to ensure that any exceptions raised in the tasks are collected
    ASAP.
    '''

    # Personal note: I consider this an antipattern, as it relies on the use of
    # unowned resources. The setup function dumps some stuff into the event
    # loop where it just whirls in the ether without a well defined owner or
    # lifetime. For this reason, there's a good chance I'll remove the
    # forever=True feature from autoasync at some point in the future.
    thing = coro(*args, **kwargs)
    if iscoroutine(thing):
        loop.create_task(thing)