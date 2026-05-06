def _handle_stop_dag(self, request):
        """ The handler for the stop_dag request.

        The stop_dag request adds a dag to the list of dags that should be stopped.
        The dag will then stop queueing new tasks and will eventually stop running.

        Args:
            request (Request): Reference to a request object containing the
                               incoming request. The payload has to contain the
                               following fields:
                                'name': the name of the dag that should be stopped

        Returns:
            Response: A response object containing the following fields:
                          - success: True if the dag was added successfully to the list
                                     of dags that should be stopped.
        """
        if (request.payload['name'] is not None) and \
           (request.payload['name'] not in self._stop_dags):
            self._stop_dags.append(request.payload['name'])
        return Response(success=True, uid=request.uid)