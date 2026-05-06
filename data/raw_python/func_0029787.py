def update_id(self, sequence_id=None, force=True):
        """Alter the sequence id, and all of the names and ids derived from it. This
        often needs to be don after an IntegrityError in a multiprocessing run"""
        from ..identity import ObjectNumber

        if sequence_id:
            self.sequence_id = sequence_id

        assert self.d_vid

        if self.id is None or force:
            dataset_id = ObjectNumber.parse(self.d_vid).rev(None)
            self.d_id = str(dataset_id)
            self.id = str(TableNumber(dataset_id, self.sequence_id))

        if self.vid is None or force:
            dataset_vid = ObjectNumber.parse(self.d_vid)
            self.vid = str(TableNumber(dataset_vid, self.sequence_id))