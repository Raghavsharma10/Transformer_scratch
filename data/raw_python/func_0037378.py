def update_metadata(self, archive_name, archive_metadata):
        '''
        Update metadata for archive ``archive_name``
        '''

        required_metadata_keys = self.required_archive_metadata.keys()
        for key, val in archive_metadata.items():
            if key in required_metadata_keys and val is None:
                raise ValueError(
                    'Cannot remove required metadata attribute "{}"'.format(
                        key))

        self._update_metadata(archive_name, archive_metadata)