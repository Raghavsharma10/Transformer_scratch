def worker_start(obj, queues, name, celery_args):
    """ Start a worker process.

    \b
    CELERY_ARGS: Additional Celery worker command line arguments.
    """
    try:
        start_worker(queues=queues.split(','),
                     config=obj['config'],
                     name=name,
                     celery_args=celery_args)
    except DataStoreNotConnected:
        click.echo(_style(obj['show_color'],
                          'Cannot connect to the Data Store server. Is the server running?',
                          fg='red', bold=True))