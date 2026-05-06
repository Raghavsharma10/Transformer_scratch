def get_plugins() -> Dict[str, pkg_resources.EntryPoint]:
        """
        Get all available plugins for unidown.

        :return: plugin name list
        :rtype: Dict[str, ~pkg_resources.EntryPoint]
        """
        return {entry.name: entry for entry in pkg_resources.iter_entry_points('unidown.plugin')}