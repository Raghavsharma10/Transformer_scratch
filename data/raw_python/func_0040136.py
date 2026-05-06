def print_config(self):
        """Print all yala configurations, including default and user's."""
        linters = self.user_linters or list(self.linters)
        print('linters:', ', '.join(linters))
        for key, value in self._config.items():
            if key != 'linters':
                print('{}: {}'.format(key, value))