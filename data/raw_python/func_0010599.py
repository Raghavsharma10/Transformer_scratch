def join_dags(self, names=None):
        """ Wait for the specified dags to terminate.

        This function blocks until the specified dags terminate. If no dags are specified
        wait for all dags of the workflow, except the dag of the task calling this signal,
        to terminate.

        Args:
            names (list): The names of the dags that have to terminate.

        Returns:
            bool: True if all the signal was sent successfully.
        """
        return self._client.send(
            Request(
                action='join_dags',
                payload={'names': names}
            )
        ).success