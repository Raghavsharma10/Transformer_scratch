def get_enabled_hook_plugins(self, hook, args, **kwargs):
        """Get enabled plugins for specified hook name.

        """
        manager = self.hook_managers[hook]
        if len(list(manager)) == 0:
            return []
        return [
            plugin for plugin in manager.map(
                self._create_hook_plugin, args, **kwargs)
            if plugin is not None
        ]