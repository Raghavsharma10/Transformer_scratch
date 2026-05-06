def check_for_stalled_tasks():
    """Check for tasks that are no longer sending a heartbeat
    """
    from api.models.tasks import Task
    for task in Task.objects.filter(status_is_running=True):
        if not task.is_responsive():
            task.system_error()
        if task.is_timed_out():
            task.timeout_error()