def command_with_options(self):
        """Add arguments from config to :attr:`command`."""
        if 'args' in self.config:
            return ' '.join((self.command, self.config['args']))
        return self.command