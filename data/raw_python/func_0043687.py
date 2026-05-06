def init_app(self, app):
        """Initialize actions with the app or blueprint.

        :param app: the Flask application or blueprint object
        :type app: :class:`~flask.Flask` or :class:`~flask.Blueprint`

        Examples::

            api = Api()
            api.add_resource(...)
            api.init_app(blueprint)
        """
        try:
            # Assume this is a blueprint and defer initialization
            if app._got_registered_once is True:
                raise ValueError("""Blueprint is already registered with an app.""")
            app.record(self._deferred_blueprint_init)
        except AttributeError:
            self._init_app(app)
        else:
            self.blueprint = app