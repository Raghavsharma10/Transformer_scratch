def stop_dag(self, name=None):
        """ Send a stop signal to the specified dag or the dag that hosts this task.

        Args:
            name str: The name of the dag that should be stopped. If no name is given the
                      dag that hosts this task is stopped.

        Upon receiving the stop signal, the dag will not queue any new tasks and wait
        for running tasks to terminate.

        Returns:
            bool: True if the signal was sent successfully.
        """
        return self._client.send(
            Request(
                action='stop_dag',
                payload={'name': name if name is not None else self._dag_name}
            )
        ).success