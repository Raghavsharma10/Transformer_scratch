def sleep(seconds=0):
    """Yield control to another eligible coroutine until at least *seconds* have
    elapsed.

    *seconds* may be specified as an integer, or a float if fractional seconds
    are desired.
    """
    loop = evergreen.current.loop
    current = Fiber.current()
    assert loop.task is not current
    timer = loop.call_later(seconds, current.switch)
    try:
        loop.switch()
    finally:
        timer.cancel()