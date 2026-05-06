def _get_contents(self):
        """Create strings from glob strings."""
        def files():
            for value in super(GlobBundle, self)._get_contents():
                for path in glob.glob(value):
                    yield path
        return list(files())