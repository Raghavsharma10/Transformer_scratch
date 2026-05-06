def _queue_dag(self, name, *, data=None):
        """ Add a new dag to the queue.

        If the stop workflow flag is set, no new dag can be queued.

        Args:
            name (str): The name of the dag that should be queued.
            data (MultiTaskData): The data that should be passed on to the new dag.

        Raises:
            DagNameUnknown: If the specified dag name does not exist

        Returns:
            str: The name of the queued dag.
        """
        if self._stop_workflow:
            return None

        if name not in self._dags_blueprint:
            raise DagNameUnknown()

        new_dag = copy.deepcopy(self._dags_blueprint[name])
        new_dag.workflow_name = self.name
        self._dags_running[new_dag.name] = self._celery_app.send_task(
            JobExecPath.Dag, args=(new_dag, self._workflow_id, data),
            queue=new_dag.queue, routing_key=new_dag.queue)

        return new_dag.name