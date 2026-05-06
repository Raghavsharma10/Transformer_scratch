def insert(self, revision, index):
        """
        Insert a :class:`revision.data.Revision` at a given index.

        :param revision:
        :type revision: :class:`revision.data.Revision`
        :param index:
        :type index: int
        """
        if not isinstance(revision, Revision):
            raise InvalidArgType()

        for rev in self.revisions:
            if rev == revision:
                return self

        self.revisions.insert(index, revision)

        return self