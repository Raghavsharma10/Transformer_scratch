def reload_configuration(self, event):
        """Event triggered configuration reload"""

        if event.target == self.uniquename:
            self.log('Reloading configuration')
            self._read_config()