def create_archive(
            self,
            archive_name,
            authority_name,
            archive_path,
            versioned,
            raise_on_err=True,
            metadata=None,
            user_config=None,
            tags=None,
            helper=False):
        '''
        Create a new data archive

        Returns
        -------
        archive : object
            new :py:class:`~datafs.core.data_archive.DataArchive` object

        '''

        archive_metadata = self._create_archive_metadata(
            archive_name=archive_name,
            authority_name=authority_name,
            archive_path=archive_path,
            versioned=versioned,
            raise_on_err=raise_on_err,
            metadata=metadata,
            user_config=user_config,
            tags=tags,
            helper=helper)

        if raise_on_err:
            self._create_archive(
                archive_name,
                archive_metadata)
        else:
            self._create_if_not_exists(
                archive_name,
                archive_metadata)

        return self.get_archive(archive_name)