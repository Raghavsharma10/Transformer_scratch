def mount(self, app=None):
        """Mounts all registered routes to a bottle.py application instance.

        Args:
            app (instance): A `bottle.Bottle()` application instance.

        Returns:
            The Router instance (for chaining purposes).
        """
        for endpoint in self._routes:
            endpoint.register_app(app)

        return self