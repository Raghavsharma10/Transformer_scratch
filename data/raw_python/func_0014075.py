def _save_config(self, filename):
        """Save the current running config to the given file."""
        self.device.show('checkpoint file {}'.format(filename), raw_text=True)