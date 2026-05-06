def list_workers(config, *, filter_by_queues=None):
    """ Return a list of all available workers.

    Args:
        config (Config): Reference to the configuration object from which the
            settings are retrieved.
        filter_by_queues (list): Restrict the returned workers to workers that listen to
            at least one of the queue names in this list.

    Returns:
        list: A list of WorkerStats objects.
    """
    celery_app = create_app(config)
    worker_stats = celery_app.control.inspect().stats()
    queue_stats = celery_app.control.inspect().active_queues()

    if worker_stats is None:
        return []

    workers = []
    for name, w_stat in worker_stats.items():
        queues = [QueueStats.from_celery(q_stat) for q_stat in queue_stats[name]]

        add_worker = filter_by_queues is None
        if not add_worker:
            for queue in queues:
                if queue.name in filter_by_queues:
                    add_worker = True
                    break

        if add_worker:
            workers.append(WorkerStats.from_celery(name, w_stat, queues))

    return workers