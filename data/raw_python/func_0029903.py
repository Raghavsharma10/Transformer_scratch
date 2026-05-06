def path(self):
        """The path of the bundle source.

        Includes the revision.

        """

        # Need to do this to ensure the function produces the
        # bundle path when called from subclasses
        names = [k for k, _, _ in Name._name_parts]

        return os.path.join(self.source,
                            self._path_join(names=names, excludes=['source', 'format'], sep=self.NAME_PART_SEP),
                            *self._local_parts()
                            )