def init_app(self, app, datastore=None):
        """Initialize the application with the Social extension

        :param app: The Flask application
        :param datastore: Connection datastore instance
        """

        datastore = datastore or self.datastore

        for key, value in default_config.items():
            app.config.setdefault(key, value)

        providers = dict()

        for key, config in app.config.items():
            if not key.startswith('SOCIAL_') or config is None or key in default_config:
                continue

            suffix = key.lower().replace('social_', '')
            default_module_name = 'flask_social.providers.%s' % suffix
            module_name = config.get('module', default_module_name)
            module = import_module(module_name)
            config = update_recursive(module.config, config)

            providers[config['id']] = OAuthRemoteApp(**config)
            providers[config['id']].tokengetter(_get_token)

        state = _get_state(app, datastore, providers)

        app.register_blueprint(create_blueprint(state, __name__))
        app.extensions['social'] = state

        return state