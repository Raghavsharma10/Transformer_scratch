def _init_app(self, app):
        """Initialize actions with the given :class:`flask.Flask` object.

        :param app: The flask application object
        :type app: :class:`~flask.Flask`
        """
        for resource, urls, kwargs in self.resources:
            self._register_view(app, resource, *urls, **kwargs)