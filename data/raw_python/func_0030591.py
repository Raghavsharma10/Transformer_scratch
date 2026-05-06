def compose(self, name, *args):
        """Compose, but don't create base directory"""

        return self._compose(name, args, mkdir=False)