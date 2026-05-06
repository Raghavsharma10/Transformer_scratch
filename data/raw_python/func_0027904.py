def from_file(cls, path):

        """
        Create a text from a file.

        Args:
            path (str): The file path.
        """

        with open(path, 'r', errors='replace') as f:
            return cls(f.read())