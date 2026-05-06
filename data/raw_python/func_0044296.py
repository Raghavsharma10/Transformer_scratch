def look_source(self, sourcepath, library_paths=None):
        """
        Open a SCSS file (sourcepath) and find all involved file through
        imports.

        This will fill internal buffers ``_CHILDREN_MAP`` and ``_PARENTS_MAP``.

        Args:
            sourcepath (str): Source file path to start searching for imports.

        Keyword Arguments:
            library_paths (list): List of directory paths for libraries to
                resolve paths if resolving fails on the base source path.
                Default to None.
        """
        # Don't inspect again source that has allready be inspected as a
        # children of a previous source
        if sourcepath not in self._CHILDREN_MAP:
            with io.open(sourcepath, 'r', encoding='utf-8') as fp:
                finded_paths = self.parse(fp.read())

            children = self.resolve(sourcepath, finded_paths,
                                    library_paths=library_paths)

            # Those files that are imported by the sourcepath
            self._CHILDREN_MAP[sourcepath] = children

            # Those files that import the sourcepath
            for p in children:
                self._PARENTS_MAP[p].add(sourcepath)

            # Start recursive finding through each resolved path that has not
            # been collected yet
            for path in children:
                if path not in self._CHILDREN_MAP:
                    self.look_source(path, library_paths=library_paths)

        return