def add(self, directory, path=None) -> None:
        """Add a directory and optionally its path."""
        objecttools.valid_variable_identifier(directory)
        if path is None:
            path = directory
        setattr(self, directory, path)