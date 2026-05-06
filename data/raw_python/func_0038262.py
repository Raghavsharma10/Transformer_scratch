def get_plugin(self, plugin_type, name, version=None):
        """
        Args:
            plugin_type(str): Parent type
            name(str): Plugin name
            version(str): Plugin version

        Returns:
            :py:class:`Plugin`: Plugin, or :py:data:`None` if plugin can't be found

        Retrieve a specific plugin. ``blacklist`` and ``type_filter`` still apply.

        If ``version`` is not specified, the newest available version is returned.
        """

        if not self.loaded:
            self.load_modules()

        # pylint: disable=protected-access
        return get_plugins()[self.group]._filter(blacklist=self.blacklist,
                                                 newest_only=True,
                                                 type_filter=self.type_filter,
                                                 type=plugin_type,
                                                 name=name,
                                                 version=version)