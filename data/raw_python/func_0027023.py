def create_future(loop):
    # pragma: no cover
    """Compatibility wrapper for the loop.create_future() call introduced in
    3.5.2."""
    if hasattr(loop, 'create_future'):
        return loop.create_future()
    return asyncio.Future(loop=loop)