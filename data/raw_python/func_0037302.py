def update(
            self,
            filepath,
            cache=False,
            remove=False,
            bumpversion=None,
            prerelease=None,
            dependencies=None,
            metadata=None,
            message=None):
        '''
        Enter a new version to a DataArchive

        Parameters
        ----------

        filepath : str
            The path to the file on your local file system

        cache : bool
            Turn on caching for this archive if not already on before update

        remove : bool
            removes a file from your local directory

        bumpversion : str
            Version component to update on write if archive is versioned. Valid
            bumpversion values are 'major', 'minor', and 'patch', representing
            the three components of the strict version numbering system (e.g.
            "1.2.3"). If bumpversion is None the version number is not updated
            on write. Either bumpversion or prerelease (or both) must be a
            non-None value. If the archive is not versioned, bumpversion is
            ignored.

        prerelease : str
            Prerelease component of archive version to update on write if
            archive is versioned. Valid prerelease values are 'alpha' and
            'beta'. Either bumpversion or prerelease (or both) must be a
            non-None value. If the archive is not versioned, prerelease is
            ignored.

        metadata : dict
            Updates to archive metadata. Pass {key: None} to remove a key from
            the archive's metadata.
        '''

        if metadata is None:
            metadata = {}

        latest_version = self.get_latest_version()

        hashval = self.api.hash_file(filepath)

        checksum = hashval['checksum']
        algorithm = hashval['algorithm']

        if checksum == self.get_latest_hash():
            self.update_metadata(metadata)

            if remove and os.path.isfile(filepath):
                os.remove(filepath)

            return

        if self.versioned:
            if latest_version is None:
                latest_version = BumpableVersion()

            next_version = latest_version.bump(
                kind=bumpversion,
                prerelease=prerelease,
                inplace=False)

        else:
            next_version = None

        next_path = self.get_version_path(next_version)

        if cache:
            self.cache(next_version)

        if self.is_cached(next_version):
            self.authority.upload(filepath, next_path)
            self.api.cache.upload(filepath, next_path, remove=remove)

        else:
            self.authority.upload(filepath, next_path, remove=remove)

        self._update_manager(
            archive_metadata=metadata,
            version_metadata=dict(
                checksum=checksum,
                algorithm=algorithm,
                version=next_version,
                dependencies=dependencies,
                message=message))