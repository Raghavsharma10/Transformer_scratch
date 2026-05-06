def from_event(cls, event):
        """ Create a JobEvent object from the event dictionary returned by celery.

        Args:
            event (dict): The dictionary as returned by celery.

        Returns:
            JobEvent: A fully initialized JobEvent object.
        """
        return cls(
            uuid=event['uuid'],
            job_type=event['job_type'],
            event_type=event['type'],
            queue=event['queue'],
            hostname=event['hostname'],
            pid=event['pid'],
            name=event['name'],
            workflow_id=event['workflow_id'],
            event_time=event['time'],
            duration=event['duration']
        )