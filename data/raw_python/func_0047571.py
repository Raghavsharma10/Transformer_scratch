def get_setup(cls):
        """Get package setup."""
        try:
            with open("setup.py") as f:
                return SetupVisitor(ast.parse(f.read()))
        except IOError as e:
            LOG.warning("Couldn't open setup file: %s", e)
            return SetupVisitor(ast.parse(""))