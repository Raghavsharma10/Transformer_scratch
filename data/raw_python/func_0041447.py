def has_new_revision(self):
        """
        :return: True if new revision exists, False if not.
        :rtype: boolean
        """
        if self.history.current_revision is None:
            return self.state.revision_id is not None

        current_revision_id = self.history.current_revision.revision_id
        return self.state.revision_id != current_revision_id