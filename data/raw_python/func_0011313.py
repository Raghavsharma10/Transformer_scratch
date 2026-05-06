def command(self, name=None):
        """A decorator to add subcommands.
        """
        def decorator(f):
            self.add_command(f, name)
            return f
        return decorator