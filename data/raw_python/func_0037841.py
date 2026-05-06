def _filter(self, blacklist=None, newest_only=False, type_filter=None, **kwargs):
        """
        Args:
            blacklist(tuple): Iterable of of BlacklistEntry objects
            newest_only(bool): Only the newest version of each plugin is returned
            type(str): Plugin type to retrieve
            name(str): Plugin name to retrieve
            version(str): Plugin version to retrieve

        Returns nested dictionary of plugins

        If a blacklist is supplied, plugins are evaluated against the blacklist entries
        """

        plugins = DictWithDotNotation()
        filtered_name = kwargs.get(self._key_attr, None)

        for key, val in self._items(type_filter, filtered_name):
            plugin_blacklist = None
            skip = False

            if blacklist:

                # Assume blacklist is correct format since it is checked by PluginLoade

                plugin_blacklist = []
                for entry in blacklist:
                    if getattr(entry, self._key_attr) not in (key, None):
                        continue
                    if all(getattr(entry, attr) is None for attr in self._bl_skip_attrs):
                        if not self._skip_empty:
                            plugins[key] = None if filtered_name else self._bl_empty()
                        skip = True
                        break

                    plugin_blacklist.append(entry)

            if not skip:
                # pylint: disable=protected-access
                result = val._filter(plugin_blacklist, newest_only=newest_only, **kwargs)

                if result or not self._skip_empty:
                    plugins[key] = result

        if filtered_name:
            return plugins.get(filtered_name, None)
        return plugins