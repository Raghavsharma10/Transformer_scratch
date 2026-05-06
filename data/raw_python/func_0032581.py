def call_method_async(self, method_name_or_object, params=None):
        """
        Calls the ``method_name`` method from the given service asynchronously
        and returns a :py:class:`gemstone.client.structs.AsyncMethodCall` instance.

        :param method_name_or_object: The name of te called method or a ``MethodCall`` instance
        :param params: A list of dict representing the parameters for the request
        :return: a :py:class:`gemstone.client.structs.AsyncMethodCall` instance.
        """
        thread_pool = self._get_thread_pool()

        if isinstance(method_name_or_object, MethodCall):
            req_obj = method_name_or_object
        else:
            req_obj = MethodCall(method_name_or_object, params)

        async_result_mp = thread_pool.apply_async(self.handle_single_request, args=(req_obj,))
        return AsyncMethodCall(req_obj=req_obj, async_resp_object=async_result_mp)