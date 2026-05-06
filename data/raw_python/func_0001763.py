def register_handler(self, callable_obj, entrypoint, methods=('GET',)):
        """Register a handler callable to a specific route.

        Args:
            entrypoint (str): The uri relative path.
            methods (tuple): A tuple of valid method strings.
            callable_obj (callable): The callable object.

        Returns:
            The Router instance (for chaining purposes).

        Raises:
            RouteError, for missing routing params or invalid callable
            object type.
        """

        router_obj = Route.wrap_callable(
            uri=entrypoint,
            methods=methods,
            callable_obj=callable_obj
        )

        if router_obj.is_valid:
            self._routes.add(router_obj)
            return self
        
        raise RouteError(  # pragma: no cover
            "Missing params: methods: {} - entrypoint: {}".format(
                methods, entrypoint
            )
        )