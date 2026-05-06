def call_method(self, method):
        """
        Calls a blocking method in an executor, in order to preserve the non-blocking behaviour

        If ``method`` is a coroutine, yields from it and returns, no need to execute in
        in an executor.

        :param method: The method or coroutine to be called (with no arguments).
        :return: the result of the method call
        """
        if self._method_is_async_generator(method):
            result = yield method()
        else:
            result = yield self.executor.submit(method)
        return result