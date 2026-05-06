def init_app(self, app):
        """Initialize Flask application."""
        app.config.from_pyfile('{0}.cfg'.format(app.name), silent=True)