def ensure_exists(self):
        """Make sure the location exists."""
        if not self.context.is_directory(self.directory):
            # This can also happen when we don't have permission to one of the
            # parent directories so we'll point that out in the error message
            # when it seems applicable (so as not to confuse users).
            if self.context.have_superuser_privileges:
                msg = "The directory %s doesn't exist!"
                raise ValueError(msg % self)
            else:
                raise ValueError(compact("""
                    The directory {location} isn't accessible, most likely
                    because it doesn't exist or because of permissions. If
                    you're sure the directory exists you can use the
                    --use-sudo option.
                """, location=self))