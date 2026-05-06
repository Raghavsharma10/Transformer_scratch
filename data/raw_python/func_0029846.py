def record_to_objects(self):
        """Create config records to match the file metadata"""
        from ..util import AttrDict

        fr = self.record

        contents = fr.unpacked_contents

        if not contents:
            return

        ad = AttrDict(contents)


        # Get time that filessystem was synchronized to the File record.
        # Maybe use this to avoid overwriting configs that changed by bundle program.
        # fs_sync_time = self._dataset.config.sync[self.file_const][self.file_to_record]

        self._dataset.config.metadata.set(ad)

        self._dataset._database.commit()

        return ad