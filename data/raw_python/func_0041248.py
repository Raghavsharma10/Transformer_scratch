def checkout(self, revision_id):
        """
        :param revision_id: :class:`revision.data.Revision` ID.
        :type revision_id: str
        """
        index = 0
        found = False
        for revision in self.revisions:
            if revision.revision_id == revision_id:
                self.current_index = index
                found = True

            index += 1

        if not found:
            raise RuntimeError("")