def call(self, request_function, set_header_callback, *args, **kwargs):
        """Rate limit the call to request_function.

        :param request_function: A function call that returns an HTTP response
            object.
        :param set_header_callback: A callback function used to set the request
            headers. This callback is called after any necessary sleep time
            occurs.
        :param *args: The positional arguments to ``request_function``.
        :param **kwargs: The keyword arguments to ``request_function``.

        """
        self.delay()
        kwargs["headers"] = set_header_callback()
        response = request_function(*args, **kwargs)
        self.update(response.headers)
        return response