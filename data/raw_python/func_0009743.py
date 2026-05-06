def init_app(self, app, **kwargs):
        """ Initializes the Flask-Bouncer extension for the specified application.

        :param app: The application.
        """
        self.app = app

        self._init_extension()

        self.app.before_request(self.check_implicit_rules)

        if kwargs.get('ensure_authorization', False):
            self.app.after_request(self.check_authorization)