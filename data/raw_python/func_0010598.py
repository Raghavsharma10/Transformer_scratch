def start_dag(self, dag, *, data=None):
        """ Schedule the execution of a dag by sending a signal to the workflow.

        Args:
            dag (Dag, str): The dag object or the name of the dag that should be started.
            data (MultiTaskData): The data that should be passed on to the new dag.

        Returns:
            str: The name of the successfully started dag.
        """
        return self._client.send(
            Request(
                action='start_dag',
                payload={'name': dag.name if isinstance(dag, Dag) else dag,
                         'data': data if isinstance(data, MultiTaskData) else None}
            )
        ).payload['dag_name']