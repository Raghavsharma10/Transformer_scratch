def register_app(self, app):
        """Register the route object to a `bottle.Bottle` app instance.

        Args:
            app (instance):

        Returns:
            Route instance (for chaining purposes)
        """
        app.route(self.uri, methods=self.methods)(self.callable_obj)

        return self