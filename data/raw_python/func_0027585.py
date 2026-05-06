def init_config(self, app):
        """Initialize configuration.

        :param app: An instance of :class:`~flask.Flask`.
        """
        app.config.setdefault('REQUIREJS_BASEURL', app.static_folder)
        app.config.setdefault('COLLECT_STATIC_ROOT', app.static_folder)
        app.config.setdefault('COLLECT_STORAGE', 'flask_collect.storage.link')
        app.config.setdefault(
            'COLLECT_FILTER', partial(collect_staticroot_removal, app))
        app.config.setdefault(
            'WEBPACKEXT_PROJECT', 'invenio_assets.webpack:project')