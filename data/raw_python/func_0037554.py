def listdir(self, location, authority_name=None):
        '''
        List archive path components at a given location

        .. Note ::

            When using listdir on versioned archives, listdir will provide the
            version numbers when a full archive path is supplied as the
            location argument. This is because DataFS stores the archive path
            as a directory and the versions as the actual files when versioning
            is on.

        Parameters
        ----------

        location: str

            Path of the "directory" to search

            `location` can be a path relative to the authority root (e.g
            `/MyFiles/Data`) or can include authority as a protocol (e.g.
            `my_auth://MyFiles/Data`). If the authority is specified as a
            protocol, the `authority_name` argument is ignored.

        authority_name: str

            Name of the authority to search (optional)

            If no authority is specified, the default authority is used (if
            only one authority is attached or if
            :py:attr:`DefaultAuthorityName` is assigned).

        Returns
        -------

        list

            Archive path components that exist at the given "directory"
            location on the specified authority

        Raises
        ------

        ValueError

            A ValueError is raised if the authority is ambiguous or invalid


        '''

        authority_name, location = self._normalize_archive_name(
            location,
            authority_name=authority_name)

        if authority_name is None:
            authority_name = self.default_authority_name

        return self._authorities[authority_name].fs.listdir(location)