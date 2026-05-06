def init_app(self, app):
        """Initialize Flask application."""
        # Ensure SECRET_KEY is set.
        SECRET_KEY = app.config.get('SECRET_KEY')

        if SECRET_KEY is None:
            app.config['SECRET_KEY'] = 'CHANGE_ME'
            warnings.warn(
                'Set configuration variable SECRET_KEY with random string',
                UserWarning)