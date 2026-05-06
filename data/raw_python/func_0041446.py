def save(self, revision):
        """
        :param revision:
        :type revision: :class:`revision.data.Revision`
        """
        if not isinstance(revision, Revision):
            raise InvalidArgType()

        self.state.update(revision)