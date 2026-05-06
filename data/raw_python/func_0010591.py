def _handle_request(self, request):
        """ Handle an incoming request by forwarding it to the appropriate method.

        Args:
            request (Request): Reference to a request object containing the
                               incoming request.

        Raises:
            RequestActionUnknown: If the action specified in the request is not known.

        Returns:
            Response: A response object containing the response from the method handling
                      the request.
        """
        if request is None:
            return Response(success=False, uid=request.uid)

        action_map = {
            'start_dag': self._handle_start_dag,
            'stop_workflow': self._handle_stop_workflow,
            'join_dags': self._handle_join_dags,
            'stop_dag': self._handle_stop_dag,
            'is_dag_stopped': self._handle_is_dag_stopped
        }

        if request.action in action_map:
            return action_map[request.action](request)
        else:
            raise RequestActionUnknown()