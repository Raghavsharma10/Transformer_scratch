def current_revision(self):
        """
        :return: The current :class:`revision.data.Revision`.
        :rtype: :class:`revision.data.Revision`
        """
        if self.current_index is None:
            return None

        if len(self.revisions) > self.current_index:
            return self.revisions[self.current_index]

        return None