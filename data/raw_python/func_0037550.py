def default_versions(self, default_versions):
        '''
        Set archive default read versions

        Parameters
        ----------
        default_versions: dict
            Dictionary of archive_name, version pairs. On read/download,
            archives in this dictionary will download the specified version
            by default. Before assignment, archive_names are checked and
            normalized.
        '''

        default_versions = {
            self._normalize_archive_name(arch)[1]: v
            for arch, v in default_versions.items()}

        self._default_versions = default_versions