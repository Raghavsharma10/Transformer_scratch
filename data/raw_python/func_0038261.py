def plugins_all(self):
        """
        All resulting versions of all plugins in the group filtered by ``blacklist``

        Returns:
            dict: Nested dictionary of plugins accessible through dot-notation.

        Similar to :py:attr:`plugins`, but lowest level is a regular dictionary of
        all unfiltered plugin versions for the given plugin type and name.

        Parent types are always included.
        Child plugins will only be included if at least one valid, non-blacklisted plugin
        is available.
        """

        if not self.loaded:
            self.load_modules()

        # pylint: disable=protected-access
        return get_plugins()[self.group]._filter(blacklist=self.blacklist,
                                                 type_filter=self.type_filter)