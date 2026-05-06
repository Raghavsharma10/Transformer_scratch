def _add_request_parameters(func):
        """Adds the ratelimit and request timeout parameters to a function."""

        # The function the decorator returns
        async def decorated_func(*args, handle_ratelimit=None, max_tries=None, request_timeout=None, **kwargs):
            return await func(*args, handle_ratelimit=handle_ratelimit, max_tries=max_tries,
                              request_timeout=request_timeout, **kwargs)

        # We return the decorated func
        return decorated_func