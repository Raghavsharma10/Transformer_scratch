def update(self, archive_name, version_metadata):
        '''
        Register a new version for archive ``archive_name``

        .. note ::

            need to implement hash checking to prevent duplicate writes
        '''
        version_metadata['updated'] = self.create_timestamp()
        version_metadata['version'] = str(
            version_metadata.get('version', None))

        if version_metadata.get('message') is not None:
            version_metadata['message'] = str(version_metadata['message'])

        self._update(archive_name, version_metadata)