def update_id(self, sequence_id=None):
        """Alter the sequence id, and all of the names and ids derived from it. This
        often needs to be don after an IntegrityError in a multiprocessing run"""
        from ..identity import GeneralNumber1

        if sequence_id:
            self.sequence_id = sequence_id

        self.vid = str(GeneralNumber1('T', self.d_vid, self.sequence_id))