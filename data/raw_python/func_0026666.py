def _get_task_priority(tasks, task_priority):
    """Get the task `priority` corresponding to the given `task_priority`.

    If `task_priority` is an integer or 'None', return it.
    If `task_priority` is a str, return the priority of the task it matches.
    Otherwise, raise `ValueError`.
    """
    if task_priority is None:
        return None
    if is_integer(task_priority):
        return task_priority
    if isinstance(task_priority, basestring):
        if task_priority in tasks:
            return tasks[task_priority].priority

    raise ValueError("Unrecognized task priority '{}'".format(task_priority))