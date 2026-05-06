def init_app(self, app):
        """Initialize Flask application."""
        if self.module:
            app.config.from_object(self.module)