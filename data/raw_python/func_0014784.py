def execute_with_delay(task_function, *args, **kwargs):
    """Run a task asynchronously after at least delay_seconds
    """
    delay = kwargs.pop('delay', 0)
    if get_setting('TEST_DISABLE_ASYNC_DELAY'):
        # Delay disabled, run synchronously
        logger.debug('Running function "%s" synchronously because '\
                     'TEST_DISABLE_ASYNC_DELAY is True'
                     % task_function.__name__)
        return task_function(*args, **kwargs)

    db.connections.close_all()
    task_function.apply_async(args=args, kwargs=kwargs, countdown=delay)