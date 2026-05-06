def task(func):
    """Decorator to run the decorated function as a Task
    """
    def task_wrapper(*args, **kwargs):
        return spawn(func, *args, **kwargs)
    return task_wrapper