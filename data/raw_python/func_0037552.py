def get_archive(self, archive_name, default_version=None):
        '''
        Retrieve a data archive

        Parameters
        ----------

        archive_name: str
            Name of the archive to retrieve

        default_version: version
            str or :py:class:`~distutils.StrictVersion` giving the default
            version number to be used on read operations

        Returns
        -------
        archive: object
            New :py:class:`~datafs.core.data_archive.DataArchive` object

        Raises
        ------

        KeyError:
            A KeyError is raised when the ``archive_name`` is not found

        '''

        auth, archive_name = self._normalize_archive_name(archive_name)

        res = self.manager.get_archive(archive_name)

        if default_version is None:
            default_version = self._default_versions.get(archive_name, None)

        if (auth is not None) and (auth != res['authority_name']):
            raise ValueError(
                'Archive "{}" not found on {}.'.format(archive_name, auth) +
                ' Did you mean "{}://{}"?'.format(
                    res['authority_name'], archive_name))

        return self._ArchiveConstructor(
            api=self,
            default_version=default_version,
            **res)