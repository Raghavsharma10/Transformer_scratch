def ensure_readable(self):
        """Make sure the location exists and is readable."""
        self.ensure_exists()
        if not self.context.is_readable(self.directory):
            if self.context.have_superuser_privileges:
                msg = "The directory %s isn't readable!"
                raise ValueError(msg % self)
            else:
                raise ValueError(compact("""
                    The directory {location} isn't readable, most likely
                    because of permissions. Consider using the --use-sudo
                    option.
                """, location=self))