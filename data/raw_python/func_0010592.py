def _handle_start_dag(self, request):
        """ The handler for the start_dag request.

        The start_dag request creates a new dag and adds it to the queue.

        Args:
            request (Request): Reference to a request object containing the
                               incoming request. The payload has to contain the
                               following fields:
                                'name': the name of the dag that should be started
                                'data': the data that is passed onto the start tasks

        Returns:
            Response: A response object containing the following fields:
                          - dag_name: The name of the started dag.
        """
        dag_name = self._queue_dag(name=request.payload['name'],
                                   data=request.payload['data'])
        return Response(success=dag_name is not None, uid=request.uid,
                        payload={'dag_name': dag_name})