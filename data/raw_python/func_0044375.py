def crossplat_loop_run(coro) -> Any:
    """Cross-platform method for running a subprocess-spawning coroutine."""
    if sys.platform == 'win32':
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        loop = asyncio.ProactorEventLoop()
    else:
        loop = asyncio.new_event_loop()

    asyncio.set_event_loop(loop)
    with contextlib.closing(loop):
        return loop.run_until_complete(coro)