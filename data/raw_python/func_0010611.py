def workflow_start(obj, queue, keep_data, name, workflow_args):
    """ Send a workflow to the queue.

    \b
    NAME: The name of the workflow that should be started.
    WORKFLOW_ARGS: Workflow arguments in the form key1=value1 key2=value2.
    """
    try:
        start_workflow(name=name,
                       config=obj['config'],
                       queue=queue,
                       clear_data_store=not keep_data,
                       store_args=dict([arg.split('=', maxsplit=1)
                                        for arg in workflow_args]))
    except (WorkflowArgumentError, WorkflowImportError) as e:
        click.echo(_style(obj['show_color'],
                          'An error occurred when trying to start the workflow',
                          fg='red', bold=True))
        click.echo('{}'.format(e))
    except WorkflowDefinitionError as e:
        click.echo(_style(obj['show_color'],
                          'The graph {} in workflow {} is not a directed acyclic graph'.
                          format(e.graph_name, e.workflow_name), fg='red', bold=True))