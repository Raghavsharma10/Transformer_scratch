def filename_sanitized(self, extension, default_filename='file'):
        """Returns a filename that is safer to use on the filesystem.

        The filename will not contain a slash (nor the path separator
        for the current platform, if different), it will not
        start with a dot, and it will have the expected extension.

        No guarantees that makes it "safe enough".
        No effort is made to remove special characters;
        using this value blindly might overwrite existing files, etc.
        """

        assert extension
        assert extension[0] != '.'
        assert default_filename
        assert '.' not in default_filename
        extension = '.' + extension

        fname = self.filename_unsafe
        if fname is None:
            fname = default_filename
        fname = posixpath.basename(fname)
        fname = os.path.basename(fname)
        fname = fname.lstrip('.')
        if not fname:
            fname = default_filename
        if not fname.endswith(extension):
            fname = fname + extension
        return fname