def ensure_writable(self):
        """Make sure the directory exists and is writable."""
        self.ensure_exists()
        if not self.context.is_writable(self.directory):
            if self.context.have_superuser_privileges:
                msg = "The directory %s isn't writable!"
                raise ValueError(msg % self)
            else:
                raise ValueError(compact("""
                    The directory {location} isn't writable, most likely due
                    to permissions. Consider using the --use-sudo option.
                """, location=self))