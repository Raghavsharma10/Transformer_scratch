def create_task_info(task):
    """Create task_info field."""
    task_info = None
    if task.get('info'):
        task_info = task['info']
    else:
        task_info = task
    return task_info