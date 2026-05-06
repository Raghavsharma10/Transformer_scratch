def get_version_path(self, version=None):
        '''
        Returns a storage path for the archive and version

        If the archive is versioned, the version number is used as the file
        path and the archive path is the directory. If not, the archive path is
        used as the file path.

        Parameters
        ----------
        version : str or object
            Version number to use as file name on versioned archives (default
            latest unless ``default_version`` set)

        Examples
        --------

        .. code-block:: python

            >>> arch = DataArchive(None, 'arch', None, 'a1', versioned=False)
            >>> print(arch.get_version_path())
            a1
            >>>
            >>> ver = DataArchive(None, 'ver', None, 'a2', versioned=True)
            >>> print(ver.get_version_path('0.0.0'))
            a2/0.0
            >>>
            >>> print(ver.get_version_path('0.0.1a1'))
            a2/0.0.1a1
            >>>
            >>> print(ver.get_version_path('latest')) # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            AttributeError: 'NoneType' object has no attribute 'manager'

        '''

        version = _process_version(self, version)

        if self.versioned:
            return fs.path.join(self.archive_path, str(version))

        else:
            return self.archive_path