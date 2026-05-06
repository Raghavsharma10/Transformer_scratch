def init_app(self, app,
                 entry_point_group_mappings='invenio_search.mappings',
                 entry_point_group_templates='invenio_search.templates',
                 **kwargs):
        """Flask application initialization.

        :param app: An instance of :class:`~flask.app.Flask`.
        """
        self.init_config(app)

        app.cli.add_command(index_cmd)

        state = _SearchState(
            app,
            entry_point_group_mappings=entry_point_group_mappings,
            entry_point_group_templates=entry_point_group_templates,
            **kwargs
        )
        self._state = app.extensions['invenio-search'] = state