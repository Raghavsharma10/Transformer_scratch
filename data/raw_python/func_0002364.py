def init_app(self, app):
        """
        Register this extension with the flask app

        :param app: A flask application
        """
        # Save this so we can use it later in the extension
        if not hasattr(app, 'extensions'):   # pragma: no cover
            app.extensions = {}
        app.extensions['flask-jwt-simple'] = self

        # Set all the default configurations for this extension
        self._set_default_configuration_options(app)
        self._set_error_handler_callbacks(app)

        # Set propagate exceptions, so all of our error handlers properly
        # work in production
        app.config['PROPAGATE_EXCEPTIONS'] = True