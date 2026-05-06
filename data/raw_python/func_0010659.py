def start_worker(queues, config, *, name=None, celery_args=None, check_datastore=True):
    """ Start a worker process.

    Args:
        queues (list): List of queue names this worker accepts jobs from.
        config (Config): Reference to the configuration object from which the
            settings for the worker are retrieved.
        name (string): Unique name for the worker. The hostname template variables from
            Celery can be used. If not given, a unique name is created.
        celery_args (list): List of additional Celery worker command line arguments.
            Please note that this depends on the version of Celery used and might change.
            Use with caution.
        check_datastore (bool): Set to True to check whether the data store is available
            prior to starting the worker.
    """
    celery_app = create_app(config)

    if check_datastore:
        with DataStore(**config.data_store,
                       auto_connect=True, handle_reconnect=False) as ds:
            celery_app.user_options['datastore_info'] = ds.server_info

    argv = [
        'worker',
        '-n={}'.format(uuid4() if name is None else name),
        '--queues={}'.format(','.join(queues))
    ]

    argv.extend(celery_args or [])

    celery_app.steps['consumer'].add(WorkerLifecycle)
    celery_app.user_options['config'] = config
    celery_app.worker_main(argv)