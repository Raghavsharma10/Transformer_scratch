def has_commit(self):
        """
        :return:
        :rtype: boolean
        """
        current_revision = self.history.current_revision
        revision_id = self.state.revision_id

        return current_revision.revision_id != revision_id