def find_author(self):
        """Get the author information from the version control system."""
        return Author(name=self.context.capture('git', 'config', 'user.name', check=False, silent=True),
                      email=self.context.capture('git', 'config', 'user.email', check=False, silent=True))