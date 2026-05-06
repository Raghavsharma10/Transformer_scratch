def _config_changed(self, data, stat):
        """Called when config changes."""

        self.config = json.loads(data)

        if self.cb:
            self.cb(self.config)