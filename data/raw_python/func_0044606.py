def get_destination(self, filepath, targetdir=None):
        """
        Return destination path from given source file path.

        Destination is allways a file with extension ``.css``.

        Args:
            filepath (str): A file path. The path is allways relative to
                sources directory. If not relative, ``targetdir`` won't be
                joined.
            absolute (bool): If given will be added at beginning of file
                path.

        Returns:
            str: Destination filepath.
        """
        dst = self.change_extension(filepath, 'css')
        if targetdir:
            dst = os.path.join(targetdir, dst)
        return dst