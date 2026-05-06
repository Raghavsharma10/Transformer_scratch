def set_postbuffering_params(self, size=None, store_dir=None):
        """Sets buffering params.

        Web-proxies like nginx are "buffered", so they wait til the whole request (and its body)
        has been read, and then it sends it to the backends.

        :param int size: The size (in bytes) of the request body after which the body will
            be stored to disk (as a temporary file) instead of memory.

        :param str|unicode store_dir: Put buffered files to the specified directory. Default: TMPDIR, /tmp/

        """
        self._set_aliased('post-buffering', size)
        self._set_aliased('post-buffering-dir', store_dir)

        return self