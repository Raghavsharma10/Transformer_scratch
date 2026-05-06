def get_request_from_view_args(self, view, args, kwargs):
        """Get request object from a handler function or method. Used internally by
        ``use_args`` and ``use_kwargs``.
        """
        if len(args) > 1 and isinstance(args[1], sanic.request.Request):
            req = args[1]
        else:
            req = args[0]
        assert isinstance(
            req, sanic.request.Request
        ), "Request argument not found for handler"
        return req