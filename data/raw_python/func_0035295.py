def task_requires(*dependencies):
    """ A task decorator that ensures a distutils dependency (or a list thereof) is met
        before that task is executed.
    """
    def entangle(task):
        "Decorator wrapper."
        if not isinstance(task, tasks.Task):
            task = tasks.Task(task)

        def tool_task(*args, **kw):
            "Install requirements, then call original task."
            install_tools(dependencies)
            return task_body(*args, **kw)

        # Apply our wrapper to original task
        task_body, task.func = task.func, tool_task
        return task

    return entangle