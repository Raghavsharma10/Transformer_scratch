def init_app(self, app):
        """Flask application initialization."""
        self.init_config(app)
        self.cache = Cache(app)
        self.is_authenticated_callback = _callback_factory(
            app.config['CACHE_IS_AUTHENTICATED_CALLBACK'])
        app.extensions['invenio-cache'] = self