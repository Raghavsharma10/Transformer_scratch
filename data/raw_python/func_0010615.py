def worker_stop(obj, worker_ids):
    """ Stop running workers.

    \b
    WORKER_IDS: The IDs of the worker that should be stopped or none to stop them all.
    """
    if len(worker_ids) == 0:
        msg = 'Would you like to stop all workers?'
    else:
        msg = '\n{}\n\n{}'.format('\n'.join(worker_ids),
                                  'Would you like to stop these workers?')

    if click.confirm(msg, default=True, abort=True):
        stop_worker(obj['config'],
                    worker_ids=list(worker_ids) if len(worker_ids) > 0 else None)