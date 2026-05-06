def reload_configuration(self, event):
        """Reload the current configuration and set up everything depending on it"""

        super(EnrolManager, self).reload_configuration(event)
        self.log('Reloaded configuration.')
        self._setup()