def init_app(self, app, **kwargs):
        """Flask application initialization."""
        self.init_config(app)
        state = _AppState(app=app, cache=kwargs.get('cache'))
        app.extensions['invenio-collections'] = state
        return state