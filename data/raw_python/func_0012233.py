def save(self):
        """
        save current config to the file
        """
        with open(self._filename, "w") as f:
            self.config.write(f)