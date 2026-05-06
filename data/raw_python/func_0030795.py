def parser(cls, v):
        """Ensure that the upstream parser gets two digits. """
        return geoid.census.State.parse(str(v).zfill(2))