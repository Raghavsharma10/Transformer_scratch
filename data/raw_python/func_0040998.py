def get_settings_list(self):
        """The settings list used for building the cache id."""
        return [
            self.source,
            self.output,
            self.kwargs,
            self.post_processors,
        ]