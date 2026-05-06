def _handle_stop_workflow(self, request):
        """ The handler for the stop_workflow request.

        The stop_workflow request adds all running dags to the list of dags
        that should be stopped and prevents new dags from being started. The dags will
        then stop queueing new tasks, which will terminate the dags and in turn the
        workflow.

        Args:
            request (Request): Reference to a request object containing the
                               incoming request.

        Returns:
            Response: A response object containing the following fields:
                          - success: True if the dags were added successfully to the list
                                     of dags that should be stopped.
        """
        self._stop_workflow = True
        for name, dag in self._dags_running.items():
            if name not in self._stop_dags:
                self._stop_dags.append(name)
        return Response(success=True, uid=request.uid)