def change_extension(self, filepath, new_extension):
        """
        Change final filename extension.

        Args:
            filepath (str): A file path (relative or absolute).
            new_extension (str): New extension name (without leading dot) to
                apply.

        Returns:
            str: Filepath with new extension.
        """
        filename, ext = os.path.splitext(filepath)
        return '.'.join([filename, new_extension])