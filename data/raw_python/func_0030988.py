def execute(function, name):
    """
    Execute a task, returning a TaskResult
    """
    try:
        return TaskResult(name, True, None, function())
    except Exception as exc:
        return TaskResult(name, False, exc, None)