def isfile(self, version=None, *args, **kwargs):
        '''
        Check whether the path exists and is a file
        '''
        version = _process_version(self, version)

        path = self.get_version_path(version)
        self.authority.fs.isfile(path, *args, **kwargs)