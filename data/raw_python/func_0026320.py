def directory(self, value):
        """Normalize the value of :attr:`directory` when it's set."""
        # Normalize the value of `directory'.
        set_property(self, "directory", parse_path(value))
        # Clear the computed values of `context' and `entries'.
        clear_property(self, "context")
        clear_property(self, "entries")