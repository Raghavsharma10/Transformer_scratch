def is_stopped(self):
        """ Check whether the task received a stop signal from the workflow.

        Tasks can use the stop flag to gracefully terminate their work. This is
        particularly important for long running tasks and tasks that employ an
        infinite loop, such as trigger tasks.

        Returns:
            bool: True if the task should be stopped.
        """
        resp = self._client.send(
            Request(
                action='is_dag_stopped',
                payload={'dag_name': self._dag_name}
            )
        )
        return resp.payload['is_stopped']