def update_id(self, sequence_id=None):
        """Alter the sequence id, and all of the names and ids derived from it. This
        often needs to be done after an IntegrityError in a multiprocessing run"""

        if sequence_id:
            self.sequence_id = sequence_id

        self._set_ids(force=True)

        if self.dataset:
            self._update_names()