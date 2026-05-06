def create(
            self,
            archive_name,
            authority_name=None,
            versioned=True,
            raise_on_err=True,
            metadata=None,
            tags=None,
            helper=False):
        '''
        Create a DataFS archive

        Parameters
        ----------

        archive_name: str
            Name of the archive

        authority_name: str
            Name of the data service to use as the archive's data authority

        versioned: bool
            If true, store all versions with explicit version numbers (defualt)

        raise_on_err: bool
            Raise an error if the archive already exists (default True)

        metadata: dict
            Dictionary of additional archive metadata

        helper: bool
            If true, interactively prompt for required metadata (default False)


        '''

        authority_name, archive_name = self._normalize_archive_name(
            archive_name, authority_name=authority_name)

        if authority_name is None:
            authority_name = self.default_authority_name

        self._validate_archive_name(archive_name)

        if metadata is None:
            metadata = {}

        res = self.manager.create_archive(
            archive_name,
            authority_name,
            archive_path=archive_name,
            versioned=versioned,
            raise_on_err=raise_on_err,
            metadata=metadata,
            user_config=self.user_config,
            tags=tags,
            helper=helper)

        return self._ArchiveConstructor(
            api=self,
            **res)