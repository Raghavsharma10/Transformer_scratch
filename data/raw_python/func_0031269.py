def init_app(self, app, entry_point_group=None, register_blueprint=True,
                 register_config_blueprint=None):
        """Flask application initialization.

        :param app: The Flask application.
        :param entry_point_group: The group entry point to load extensions.
            (Default: ``invenio_jsonschemas.schemas``)
        :param register_blueprint: Register the blueprints.
        :param register_config_blueprint: Register blueprint for the specific
            app from a config variable.
        """
        self.init_config(app)

        if not entry_point_group:
            entry_point_group = self.kwargs['entry_point_group'] \
                if 'entry_point_group' in self.kwargs \
                else 'invenio_jsonschemas.schemas'

        state = InvenioJSONSchemasState(app)

        # Load the json-schemas from extension points.
        if entry_point_group:
            for base_entry in pkg_resources.iter_entry_points(
                    entry_point_group):
                directory = os.path.dirname(base_entry.load().__file__)
                state.register_schemas_dir(directory)

        # Init blueprints
        _register_blueprint = app.config.get(register_config_blueprint)
        if _register_blueprint is not None:
            register_blueprint = _register_blueprint

        if register_blueprint:
            app.register_blueprint(
                create_blueprint(state),
                url_prefix=app.config['JSONSCHEMAS_ENDPOINT']
            )

        self._state = app.extensions['invenio-jsonschemas'] = state
        return state