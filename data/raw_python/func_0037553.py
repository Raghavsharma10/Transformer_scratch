def batch_get_archive(self, archive_names, default_versions=None):
        '''
        Batch version of :py:meth:`~DataAPI.get_archive`

        Parameters
        ----------

        archive_names: list

            Iterable of archive names to retrieve

        default_versions: str, object, or dict

            Default versions to assign to each returned archive. May be a dict
            with archive names as keys and versions as values, or may be a
            version, in which case the same version is used for all archives.
            Versions must be a strict version number string, a
            :py:class:`~distutils.version.StrictVersion`, or a
            :py:class:`~datafs.core.versions.BumpableVersion` object.

        Returns
        -------

        archives: list

            List of :py:class:`~datafs.core.data_archive.DataArchive` objects.
            If an archive is not found, it is omitted (``batch_get_archive``
            does not raise a ``KeyError`` on invalid archive names).

        '''

        # toss prefixes and normalize names
        archive_names = map(
            lambda arch: self._normalize_archive_name(arch)[1],
            archive_names)

        responses = self.manager.batch_get_archive(archive_names)

        archives = {}

        if default_versions is None:
            default_versions = {}

        for res in responses:
            res['archive_name'] = self._normalize_archive_name(
                res['archive_name'])

            archive_name = res['archive_name']

            if hasattr(default_versions, 'get'):

                # Get version number from default_versions or
                # self._default_versions if key not present.
                default_version = default_versions.get(
                    archive_name,
                    self._default_versions.get(archive_name, None))

            else:
                default_version = default_versions

            archive = self._ArchiveConstructor(
                api=self,
                default_version=default_version,
                **res)

            archives[archive_name] = archive

        return archives