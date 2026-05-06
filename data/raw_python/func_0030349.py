def start(
        loop: abstract_loop = None,
        interval: float = 0.5,
        hook: hook_type = None) -> asyncio.Task:
    """
    Start the reloader.

    Create the task which is watching loaded modules
    and manually added files via ``watch()``
    and reloading the process in case of modification.
    Attach this task to the loop.

    If ``hook`` is provided, it will be called right before
    the application goes to the reload stage.
    """
    if loop is None:
        loop = asyncio.get_event_loop()

    global reload_hook
    if hook is not None:
        reload_hook = hook

    global task
    if not task:
        modify_times = {}
        executor = ThreadPoolExecutor(1)
        task = call_periodically(
            loop,
            interval,
            check_and_reload,
            modify_times,
            executor,
        )
    return task