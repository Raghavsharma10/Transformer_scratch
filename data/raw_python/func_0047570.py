def get_requirements(cls):
        """Get package requirements."""
        try:
            with open("requirements.txt") as f:
                return tuple(parse_requirements(f.readlines()))
        except IOError as e:
            LOG.debug("Couldn't open requirements file: %s", e)
            return ()