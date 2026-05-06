def update(self, settings):
        """
        Update object attributes from given settings

        Args:
            settings (dict): Dictionnary of elements to update settings.

        Returns:
            dict: Dictionnary of all current saved settings.

        """
        settings = self.clean(settings)

        # Update internal dict
        self._settings.update(settings)

        # Push every setting items as class object attributes
        self.set_settings(settings)

        return self._settings