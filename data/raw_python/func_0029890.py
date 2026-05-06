def init_app(self, app):
        """
        Find and configure the user database from specified file
        """
        app.config.setdefault('FLASK_AUTH_ALL', False)
        app.config.setdefault('FLASK_AUTH_REALM', 'Login Required')
        # Default set to bad file to trigger IOError
        app.config.setdefault('FLASK_HTPASSWD_PATH', '/^^^/^^^')

        # Load up user database
        try:
            self.load_users(app)
        except IOError:
            log.critical(
                'No htpasswd file loaded, please set `FLASK_HTPASSWD`'
                'or `FLASK_HTPASSWD_PATH` environment variable to a '
                'valid apache htpasswd file.'
            )

        # Allow requiring auth for entire app, with pre request method
        @app.before_request
        def require_auth():  # pylint: disable=unused-variable
            """Pre request processing for enabling full app authentication."""
            if not current_app.config['FLASK_AUTH_ALL']:
                return
            is_valid, user = self.authenticate()
            if not is_valid:
                return self.auth_failed()
            g.user = user