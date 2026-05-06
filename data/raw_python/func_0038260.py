def plugins(self):
        """
        Newest version of all plugins in the group filtered by ``blacklist``

        Returns:
            dict: Nested dictionary of plugins accessible through dot-notation.

        Plugins are returned in a nested dictionary, but can also be accessed through dot-notion.
        Just as when accessing an undefined dictionary key with index-notation,
        a :py:exc:`KeyError` will be raised if the plugin type or plugin does not exist.

        Parent types are always included.
        Child plugins will only be included if a valid, non-blacklisted plugin is available.
        """

        if not self.loaded:
            self.load_modules()

        # pylint: disable=protected-access
        return get_plugins()[self.group]._filter(blacklist=self.blacklist, newest_only=True,
                                                 type_filter=self.type_filter)