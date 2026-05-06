def is_partial(self, filepath):
        """
        Check if file is a Sass partial source (see
        `Sass partials Reference`_).

        Args:
            filepath (str): A file path. Can be absolute, relative or just a
            filename.

        Returns:
            bool: True if file is a partial source, else False.
        """
        path, filename = os.path.split(filepath)
        return filename.startswith('_')