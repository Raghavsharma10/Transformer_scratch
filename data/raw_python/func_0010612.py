def workflow_stop(obj, names):
    """ Stop one or more running workflows.

    \b
    NAMES: The names, ids or job ids of the workflows that should be stopped.
           Leave empty to stop all running workflows.
    """
    if len(names) == 0:
        msg = 'Would you like to stop all workflows?'
    else:
        msg = '\n{}\n\n{}'.format('\n'.join(names),
                                  'Would you like to stop these jobs?')

    if click.confirm(msg, default=True, abort=True):
        stop_workflow(obj['config'], names=names if len(names) > 0 else None)