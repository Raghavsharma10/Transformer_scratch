def set_settings(self, settings):
        """
        Set every given settings as object attributes.

        Args:
            settings (dict): Dictionnary of settings.

        """
        for k, v in settings.items():
            setattr(self, k, v)