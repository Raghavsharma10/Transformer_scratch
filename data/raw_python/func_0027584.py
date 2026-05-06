def init_app(self, app, entry_point_group='invenio_assets.bundles',
                 **kwargs):
        """Initialize application object.

        :param app: An instance of :class:`~flask.Flask`.
        :param entry_point_group: A name of entry point group used to load
            ``webassets`` bundles.

        .. versionchanged:: 1.0.0b2
           The *entrypoint* has been renamed to *entry_point_group*.
        """
        self.init_config(app)
        self.env.init_app(app)
        self.collect.init_app(app)
        self.webpack.init_app(app)

        if entry_point_group:
            self.load_entrypoint(entry_point_group)

        app.extensions['invenio-assets'] = self