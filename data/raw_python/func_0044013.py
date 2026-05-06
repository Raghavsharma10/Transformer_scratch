def clean(self, settings):
        """
        Filter given settings to keep only key names available in
        ``DEFAULT_SETTINGS``.

        Args:
            settings (dict): Loaded settings.

        Returns:
            dict: Settings object filtered.

        """
        return {k: v for k, v in settings.items() if k in DEFAULT_SETTINGS}