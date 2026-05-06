def init_app(self, app):
        """Flask application initialization."""
        self.init_config(app)
        app.cli.add_command(openaire)
        before_record_index.connect(indexer_receiver, sender=app)
        app.extensions['invenio-openaire'] = self