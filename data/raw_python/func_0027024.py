def create_task(coro, loop):
    # pragma: no cover
    """Compatibility wrapper for the loop.create_task() call introduced in
    3.4.2."""
    if hasattr(loop, 'create_task'):
        return loop.create_task(coro)
    return asyncio.Task(coro, loop=loop)