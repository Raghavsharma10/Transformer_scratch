def to_dict(self):
        """ Return a dictionary of the worker stats.

        Returns:
            dict: Dictionary of the stats.
        """
        return {
            'name': self.name,
            'broker': self.broker.to_dict(),
            'pid': self.pid,
            'process_pids': self.process_pids,
            'concurrency': self.concurrency,
            'job_count': self.job_count,
            'queues': [q.to_dict() for q in self.queues]
        }