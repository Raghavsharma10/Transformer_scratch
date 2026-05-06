def bind(self, flask_app):
        """Bind the service API urls to a flask app."""
        self.metadata.bind(flask_app, self.name, self.group)