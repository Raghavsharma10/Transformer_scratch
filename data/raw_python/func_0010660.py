def stop_worker(config, *, worker_ids=None):
    """ Stop a worker process.

    Args:
        config (Config): Reference to the configuration object from which the
            settings for the worker are retrieved.
        worker_ids (list): An optional list of ids for the worker that should be stopped.
    """
    if worker_ids is not None and not isinstance(worker_ids, list):
        worker_ids = [worker_ids]

    celery_app = create_app(config)
    celery_app.control.shutdown(destination=worker_ids)