def _patch_expand_paths(self, settings, name, value):
        """
        Apply ``SettingsPostProcessor._patch_expand_path`` to each element in
        list.

        Args:
            settings (dict): Current settings.
            name (str): Setting name.
            value (list): List of paths to patch.

        Returns:
            list: Patched path list to an absolute path.

        """
        return [self._patch_expand_path(settings, name, item)
                for item in value]