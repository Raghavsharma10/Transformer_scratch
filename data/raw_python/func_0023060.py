def route_request(self, request_json, metadata=None):

        ''' Route the request object to the right handler function '''
        request = Request(request_json)
        request.metadata = metadata
        # add reprompt handler or some such for default?
        handler_fn = self._handlers[self._default] # Set default handling for noisy requests

        if not request.is_intent() and (request.request_type() in self._handlers):
            '''  Route request to a non intent handler '''
            handler_fn = self._handlers[request.request_type()]

        elif request.is_intent() and request.intent_name() in self._handlers['IntentRequest']:
            ''' Route to right intent handler '''
            handler_fn = self._handlers['IntentRequest'][request.intent_name()]

        response = handler_fn(request)
        response.set_session(request.session)
        return response.to_json()