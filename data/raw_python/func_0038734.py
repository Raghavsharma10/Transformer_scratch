def spawn(func, *args, **kwargs):
    """Create a task to run ``func(*args, **kwargs)``.  Returns a
    :class:`Task` objec.

    Execution control returns immediately to the caller; the created task
    is merely scheduled to be run at the next available opportunity.
    Use :func:`spawn_later` to  arrange for tasks to be spawned
    after a finite delay.
    """
    t = Task(target=func, args=args, kwargs=kwargs)
    t.start()
    return t