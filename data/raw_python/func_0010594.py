def _handle_join_dags(self, request):
        """ The handler for the join_dags request.

        If dag names are given in the payload only return a valid Response if none of
        the dags specified by the names are running anymore. If no dag names are given,
        wait for all dags except one, which by design is the one that issued the request,
        to be finished.

        Args:
            request (Request): Reference to a request object containing the
                               incoming request.

        Returns:
            Response: A response object containing the following fields:
                          - success: True if all dags the request was waiting for have
                                     completed.
        """
        if request.payload['names'] is None:
            send_response = len(self._dags_running) <= 1
        else:
            send_response = all([name not in self._dags_running.keys()
                                 for name in request.payload['names']])

        if send_response:
            return Response(success=True, uid=request.uid)
        else:
            return None