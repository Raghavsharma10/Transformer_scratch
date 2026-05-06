def chain_tasks(tasks):
    """
    Chain given tasks. Set each task to run after its previous task.

    :param tasks: Tasks list.

    :return: Given tasks list.
    """
    # If given tasks list is not empty
    if tasks:
        # Previous task
        previous_task = None

        # For given tasks list's each task
        for task in tasks:
            # If the task is not None.
            # Task can be None to allow code like ``task if _PY2 else None``.
            if task is not None:
                # If previous task is not None
                if previous_task is not None:
                    # Set the task to run after the previous task
                    task.set_run_after(previous_task)

                # Set the task as previous task for the next task
                previous_task = task

    # Return given tasks list.
    return tasks