def to_dict(self):
        """ Return a dictionary of the job stats.

        Returns:
            dict: Dictionary of the stats.
        """
        return {
            'name': self.name,
            'id': self.id,
            'type': self.type,
            'workflow_id': self.workflow_id,
            'queue': self.queue,
            'start_time': self.start_time,
            'arguments': self.arguments,
            'acknowledged': self.acknowledged,
            'func_name': self.func_name,
            'hostname': self.hostname,
            'worker_name': self.worker_name,
            'worker_pid': self.worker_pid,
            'routing_key': self.routing_key
        }