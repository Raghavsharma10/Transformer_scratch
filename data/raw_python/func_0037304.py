def open(
            self,
            mode='r',
            version=None,
            bumpversion=None,
            prerelease=None,
            dependencies=None,
            metadata=None,
            message=None,
            *args,
            **kwargs):
        '''
        Opens a file for read/write

        Parameters
        ----------
        mode : str
            Specifies the mode in which the file is opened (default 'r')

        version : str
            Version number of the file to open (default latest)

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


        args, kwargs sent to file system opener

        '''

        if metadata is None:
            metadata = {}

        latest_version = self.get_latest_version()
        version = _process_version(self, version)

        version_hash = self.get_version_hash(version)

        if self.versioned:

            if latest_version is None:
                latest_version = BumpableVersion()

            next_version = latest_version.bump(
                kind=bumpversion,
                prerelease=prerelease,
                inplace=False)

            msg = "Version must be bumped on write. " \
                "Provide bumpversion and/or prerelease."

            assert next_version > latest_version, msg

            read_path = self.get_version_path(version)
            write_path = self.get_version_path(next_version)

        else:
            read_path = self.archive_path
            write_path = self.archive_path
            next_version = None

        # version_check returns true if fp's hash is current as of read
        def version_check(chk):
            return chk['checksum'] == version_hash

        # Updater updates the manager with the latest version number
        def updater(checksum, algorithm):
            self._update_manager(
                archive_metadata=metadata,
                version_metadata=dict(
                    version=next_version,
                    dependencies=dependencies,
                    checksum=checksum,
                    algorithm=algorithm,
                    message=message))

        opener = data_file.open_file(
            self.authority,
            self.api.cache,
            updater,
            version_check,
            self.api.hash_file,
            read_path,
            write_path,
            mode=mode,
            *args,
            **kwargs)

        with opener as f:
            yield f