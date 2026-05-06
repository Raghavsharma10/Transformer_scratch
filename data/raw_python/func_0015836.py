def _open(self, mode='r'):
        """Open the password file in the specified mode
        """
        open_file = None
        writeable = 'w' in mode or 'a' in mode or '+' in mode
        try:
            # NOTE: currently the MemOpener does not split off any filename
            #       which causes errors on close()
            #       so we add a dummy name and open it separately
            if (self.filename.startswith('mem://')
                    or self.filename.startswith('ram://')):
                open_file = fs.opener.fsopendir(self.filename).open('kr.cfg',
                                                                    mode)
            else:
                if not hasattr(self, '_pyfs'):
                    # reuse the pyfilesystem and path
                    self._pyfs, self._path = fs.opener.opener.parse(
                        self.filename, writeable=writeable)
                    # cache if permitted
                    if self._cache_timeout is not None:
                        self._pyfs = fs.remote.CacheFS(
                            self._pyfs, cache_timeout=self._cache_timeout)
                open_file = self._pyfs.open(self._path, mode)
        except fs.errors.ResourceNotFoundError:
            if self._can_create:
                segments = fs.opener.opener.split_segments(self.filename)
                if segments:
                    # this seems broken, but pyfilesystem uses it, so we must
                    fs_name, credentials, url1, url2, path = segments.groups()
                    assert fs_name, 'Should be a remote filesystem'
                    host = ''
                    # allow for domain:port
                    if ':' in url2:
                        split_url2 = url2.split('/', 1)
                        if len(split_url2) > 1:
                            url2 = split_url2[1]
                        else:
                            url2 = ''
                        host = split_url2[0]
                    pyfs = fs.opener.opener.opendir(
                        '%s://%s' % (fs_name, host))
                    # cache if permitted
                    if self._cache_timeout is not None:
                        pyfs = fs.remote.CacheFS(
                            pyfs, cache_timeout=self._cache_timeout)
                    # NOTE: fs.path.split does not function in the same
                    # way os os.path.split... at least under windows
                    url2_path, url2_filename = os.path.split(url2)
                    if url2_path and not pyfs.exists(url2_path):
                        pyfs.makedir(url2_path, recursive=True)
                else:
                    # assume local filesystem
                    full_url = fs.opener._expand_syspath(self.filename)
                    # NOTE: fs.path.split does not function in the same
                    # way os os.path.split... at least under windows
                    url2_path, url2 = os.path.split(full_url)
                    pyfs = fs.osfs.OSFS(url2_path)

                try:
                    # reuse the pyfilesystem and path
                    self._pyfs = pyfs
                    self._path = url2
                    return pyfs.open(url2, mode)
                except fs.errors.ResourceNotFoundError:
                    if writeable:
                        raise
                    else:
                        pass
            # NOTE: ignore read errors as the underlying caller can fail safely
            if writeable:
                raise
            else:
                pass
        return open_file