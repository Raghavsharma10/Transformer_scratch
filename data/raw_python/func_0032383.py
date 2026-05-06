def fetch(
            self,
            url,
            filename=None,
            decompress=False,
            force=False,
            timeout=None,
            use_wget_if_available=True):
        """
        Return the local path to the downloaded copy of a given URL.
        Don't download the file again if it's already present,
        unless `force` is True.
        """
        key = (url, decompress)
        if not force and key in self._local_paths:
            path = self._local_paths[key]
            if exists(path):
                return path
            else:
                del self._local_paths[key]
        path = download.fetch_file(
            url,
            filename=filename,
            decompress=decompress,
            subdir=self.subdir,
            force=force,
            timeout=timeout,
            use_wget_if_available=use_wget_if_available)

        self._local_paths[key] = path
        return path