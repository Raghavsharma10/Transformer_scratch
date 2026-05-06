def download(self, path, progress_callback=None, chunk_size=1024**2):
        """
        Download the export archive.

        .. warning::

            If you pass this function an open file-like object as the ``path``
            parameter, the function will not close that file for you.

        If a ``path`` parameter is a directory, this function will use the
        Export name to determine the name of the file (returned). If the
        calculated download file path already exists, this function will raise
        a DownloadError.

        You can also specify the filename as a string. This will be passed to
        the built-in :func:`open` and we will read the content into the file.

        Instead, if you want to manage the file object yourself, you need to
        provide either a :class:`io.BytesIO` object or a file opened with the
        `'b'` flag. See the two examples below for more details.

        :param path: Either a string with the path to the location
            to save the response content, or a file-like object expecting bytes.
        :param function progress_callback: An optional callback
                function which receives upload progress notifications. The function should take two
                arguments: the number of bytes recieved, and the total number of bytes to recieve.
        :param int chunk_size: Chunk size in bytes for streaming large downloads and progress reporting. 1MB by default
        :returns The name of the automatic filename that would be used.
        :rtype: str
        """
        if not self.download_url or self.state != 'complete':
            raise DownloadError("Download not available")

        # ignore parsing the Content-Disposition header, since we know the name
        download_filename = "{}.zip".format(self.name)
        fd = None

        if isinstance(getattr(path, 'write', None), collections.Callable):
            # already open file-like object
            fd = path
        elif os.path.isdir(path):
            # directory to download to, using the export name
            path = os.path.join(path, download_filename)
            # do not allow overwriting
            if os.path.exists(path):
                raise DownloadError("Download file already exists: %s" % path)
        elif path:
            # fully qualified file path
            # allow overwriting
            pass
        elif not path:
            raise DownloadError("Empty download file path")

        with contextlib.ExitStack() as stack:
            if not fd:
                fd = open(path, 'wb')
                # only close a file we open
                stack.callback(fd.close)

            r = self._manager.client.request('GET', self.download_url, stream=True)
            stack.callback(r.close)

            bytes_written = 0
            try:
                bytes_total = int(r.headers.get('content-length', None))
            except TypeError:
                bytes_total = None

            if progress_callback:
                # initial callback (0%)
                progress_callback(bytes_written, bytes_total)

            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)
                bytes_written += len(chunk)
                if progress_callback:
                    progress_callback(bytes_written, bytes_total)

        return download_filename