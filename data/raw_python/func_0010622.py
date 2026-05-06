def _cleanup_workflow(config, task_id, args, **kwargs):
    """ Cleanup the results of a workflow when it finished.

    Connects to the postrun signal of Celery. If the signal was sent by a workflow,
    remove the result from the result backend.

    Args:
        task_id (str): The id of the task.
        args (tuple): The arguments the task was started with.
        **kwargs: Keyword arguments from the hook.
    """
    from lightflow.models import Workflow
    if isinstance(args[0], Workflow):
        if config.celery['result_expires'] == 0:
            AsyncResult(task_id).forget()