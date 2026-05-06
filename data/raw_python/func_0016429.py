def shared_limit(self, limit_value, scope, key_func=None,
                     error_message=None, exempt_when=None):
        """
        decorator to be applied to multiple routes sharing the same rate limit.

        :param limit_value: rate limit string or a callable that returns a string.
         :ref:`ratelimit-string` for more details.
        :param scope: a string or callable that returns a string
         for defining the rate limiting scope.
        :param function key_func: function/lambda to extract the unique identifier for
         the rate limit. defaults to remote address of the request.
        :param error_message: string (or callable that returns one) to override the
         error message used in the response.
        """
        return self.__limit_decorator(
            limit_value, key_func, True, scope, error_message=error_message,
            exempt_when=exempt_when
        )