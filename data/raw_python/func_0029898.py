def source_path(self):
        """The name in a form suitable for use in a filesystem.

        Excludes the revision

        """
        # Need to do this to ensure the function produces the
        # bundle path when called from subclasses
        names = [k for k, _, _ in self._name_parts]

        parts = [self.source]

        if self.bspace:
            parts.append(self.bspace)

        parts.append(
            self._path_join(names=names, excludes=['source', 'version', 'bspace'], sep=self.NAME_PART_SEP))

        return os.path.join(*parts)