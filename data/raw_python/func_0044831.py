def candidate_paths(self, filepath):
        """
        Return candidates path for given path

        * If Filename does not starts with ``_``, will build a candidate for
          both with and without ``_`` prefix;
        * Will build For each available extensions if filename does not have
          an explicit extension;
        * Leading path directory is preserved;

        Args:
            filepath (str): Relative path as finded in an import rule from a
                SCSS source.

        Returns:
            list: Builded candidate paths (as relative paths).
        """
        filelead, filetail = os.path.split(filepath)
        name, extension = os.path.splitext(filetail)
        # Removed leading dot from extension
        if extension:
            extension = extension[1:]

        filenames = [name]
        # If underscore prefix is present, dont need to double underscore
        if not name.startswith('_'):
            filenames.append("_{}".format(name))

        # If explicit extension, dont need to add more candidate extensions
        if extension and extension in self.CANDIDATE_EXTENSIONS:
            filenames = [".".join([k, extension]) for k in filenames]
        # Else if no extension or not candidate, add candidate extensions
        else:
            # Restore uncandidate extensions if any
            if extension:
                filenames = [".".join([k, extension]) for k in filenames]
            new = []
            for ext in self.CANDIDATE_EXTENSIONS:
                new.extend([".".join([k, ext]) for k in filenames])
            filenames = new

        # Return candidates with restored leading path if any
        return [os.path.join(filelead, v) for v in filenames]