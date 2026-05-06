def _write_config(self):
        """Write this component's configuration back to the database"""

        if not self.config:
            self.log("Unable to write non existing configuration", lvl=error)
            return

        self.config.save()
        self.log("Configuration stored.")