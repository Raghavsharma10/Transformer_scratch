def _handle_is_dag_stopped(self, request):
        """ The handler for the dag_stopped request.

        The dag_stopped request checks whether a dag is flagged to be terminated.

        Args:
            request (Request): Reference to a request object containing the
                               incoming request. The payload has to contain the
                               following fields:
                                'dag_name': the name of the dag that should be checked

        Returns:
            Response: A response object containing the following fields:
                          - is_stopped: True if the dag is flagged to be stopped.
        """
        return Response(success=True,
                        uid=request.uid,
                        payload={
                            'is_stopped': request.payload['dag_name'] in self._stop_dags
                        })