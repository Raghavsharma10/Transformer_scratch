def from_celery(cls, name, worker_dict, queues):
        """ Create a WorkerStats object from the dictionary returned by celery.

        Args:
            name (str): The name of the worker.
            worker_dict (dict): The dictionary as returned by celery.
            queues (list): A list of QueueStats objects that represent the queues this
                worker is listening on.

        Returns:
            WorkerStats: A fully initialized WorkerStats object.
        """
        return WorkerStats(
            name=name,
            broker=BrokerStats.from_celery(worker_dict['broker']),
            pid=worker_dict['pid'],
            process_pids=worker_dict['pool']['processes'],
            concurrency=worker_dict['pool']['max-concurrency'],
            job_count=worker_dict['pool']['writes']['total'],
            queues=queues
        )