def limit(self, limit_value, key_func=None, per_method=False,
              methods=None, error_message=None, exempt_when=None):
        """
        decorator to be used for rate limiting individual routes.

        :param limit_value: rate limit string or a callable that returns a string.
         :ref:`ratelimit-string` for more details.
        :param function key_func: function/lambda to extract the unique identifier for
         the rate limit. defaults to remote address of the request.
        :param bool per_method: whether the limit is sub categorized into the http
         method of the request.
        :param list methods: if specified, only the methods in this list will be rate
         limited (default: None).
        :param error_message: string (or callable that returns one) to override the
         error message used in the response.
        :return:
        """
        return self.__limit_decorator(limit_value, key_func, per_method=per_method,
                                      methods=methods, error_message=error_message,
                                      exempt_when=exempt_when)