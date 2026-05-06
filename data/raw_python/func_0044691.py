def parse_filepath(self, filepath=None):
        """
        Parse given filepath to split possible path directory from filename.

        * If path directory is empty, will use ``basedir`` attribute as base
          filepath;
        * If path directory is absolute, ignore ``basedir`` attribute;
        * If path directory is relative, join it to ``basedir`` attribute;

        Keyword Arguments:
            filepath (str): Filepath to use to search for settings file. Will
                use value from ``_default_filename`` class attribute if empty.

                If filepath contain a directory path, it will be splitted from
                filename and used as base directory (and update object
                ``basedir`` attribute).

        Returns:
            tuple: Separated path directory and filename.
        """
        filepath = filepath or self._default_filename

        path, filename = os.path.split(filepath)

        if not path:
            path = self.basedir
        elif not os.path.isabs(path):
            path = os.path.join(self.basedir, path)

        return os.path.normpath(path), filename