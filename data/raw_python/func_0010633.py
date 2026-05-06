def start_workflow(name, config, *, queue=DefaultJobQueueName.Workflow,
                   clear_data_store=True, store_args=None):
    """ Start a single workflow by sending it to the workflow queue.

    Args:
        name (str): The name of the workflow that should be started. Refers to the
            name of the workflow file without the .py extension.
        config (Config): Reference to the configuration object from which the
            settings for the workflow are retrieved.
        queue (str): Name of the queue the workflow should be scheduled to.
        clear_data_store (bool): Remove any documents created during the workflow
            run in the data store after the run.
        store_args (dict): Dictionary of additional arguments that are ingested into the
            data store prior to the execution of the workflow.
    Returns:
        str: The ID of the workflow job.
    Raises:
        WorkflowArgumentError: If the workflow requires arguments to be set in store_args
            that were not supplied to the workflow.
        WorkflowImportError: If the import of the workflow fails.
    """
    try:
        wf = Workflow.from_name(name,
                                queue=queue,
                                clear_data_store=clear_data_store,
                                arguments=store_args)
    except DirectedAcyclicGraphInvalid as e:
        raise WorkflowDefinitionError(workflow_name=name,
                                      graph_name=e.graph_name)

    celery_app = create_app(config)
    result = celery_app.send_task(JobExecPath.Workflow,
                                  args=(wf,), queue=queue, routing_key=queue)
    return result.id