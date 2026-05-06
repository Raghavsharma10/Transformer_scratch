def wrap_callable(cls, uri, methods, callable_obj):
        """Wraps function-based callable_obj into a `Route` instance, else
        proxies a `bottle_neck.handlers.BaseHandler` subclass instance.

        Args:
            uri (str):  The uri relative path.
            methods (tuple): A tuple of valid method strings.
            callable_obj (instance): The callable object.

        Returns:
            A route instance.

        Raises:
            RouteError for invalid callable object type.
        """
        if isinstance(callable_obj, HandlerMeta):
            callable_obj.base_endpoint = uri
            callable_obj.is_valid = True
            return callable_obj

        if isinstance(callable_obj, types.FunctionType):
            return cls(uri=uri, methods=methods, callable_obj=callable_obj)

        raise RouteError("Invalid handler type.")