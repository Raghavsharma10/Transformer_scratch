def download(self, filepath, version=None):
        '''
        Downloads a file from authority to local path

        1. First checks in cache to check if file is there and if it is, is it
           up to date
        2. If it is not up to date, it will download the file to cache
        '''

        version = _process_version(self, version)

        dirname, filename = os.path.split(
            os.path.abspath(os.path.expanduser(filepath)))

        assert os.path.isdir(dirname), 'Directory  not found: "{}"'.format(
            dirname)

        local = OSFS(dirname)

        version_hash = self.get_version_hash(version)

        # version_check returns true if fp's hash is current as of read
        def version_check(chk):
            return chk['checksum'] == version_hash

        if os.path.exists(filepath):
            if version_check(self.api.hash_file(filepath)):
                return

        read_path = self.get_version_path(version)

        with data_file._choose_read_fs(
                self.authority,
                self.api.cache,
                read_path,
                version_check,
                self.api.hash_file) as read_fs:

            fs.utils.copyfile(
                read_fs,
                read_path,
                local,
                filename)