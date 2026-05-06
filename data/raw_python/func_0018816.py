def add_modules(self, package):
        """Add the modules of the given package without their members."""
        for name in os.listdir(package.__path__[0]):
            if name.startswith('_'):
                continue
            name = name.split('.')[0]
            short = '|%s|' % name
            long = ':mod:`~%s.%s`' % (package.__package__, name)
            self._short2long[short] = long