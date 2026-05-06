def call_method(self, method_name_or_object, params=None):
        """
        Calls the ``method_name`` method from the given service and returns a
        :py:class:`gemstone.client.structs.Result` instance.

        :param method_name_or_object: The name of te called method or a ``MethodCall`` instance
        :param params: A list of dict representing the parameters for the request
        :return: a :py:class:`gemstone.client.structs.Result` instance.
        """
        if isinstance(method_name_or_object, MethodCall):
            req_obj = method_name_or_object
        else:
            req_obj = MethodCall(method_name_or_object, params)
        raw_response = self.handle_single_request(req_obj)
        response_obj = Result(result=raw_response["result"], error=raw_response['error'],
                              id=raw_response["id"], method_call=req_obj)
        return response_obj