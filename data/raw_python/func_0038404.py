def bind(self, app):
        """Bind API to Muffin."""
        self.parent = app
        app.add_subapp(self.prefix, self.app)