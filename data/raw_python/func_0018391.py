def _download(self, link):
        """Download a file, and return its name within my temp dir.

        This does no verification of HTTPS certs, but our checking hashes
        makes that largely unimportant. It would be nice to be able to use the
        requests lib, which can verify certs, but it is guaranteed to be
        available only in pip >= 1.5.

        This also drops support for proxies and basic auth, though those could
        be added back in.

        """
        # Based on pip 1.4.1's URLOpener but with cert verification removed
        def opener(is_https):
            if is_https:
                opener = build_opener(HTTPSHandler())
                # Strip out HTTPHandler to prevent MITM spoof:
                for handler in opener.handlers:
                    if isinstance(handler, HTTPHandler):
                        opener.handlers.remove(handler)
            else:
                opener = build_opener()
            return opener

        # Descended from unpack_http_url() in pip 1.4.1
        def best_filename(link, response):
            """Return the most informative possible filename for a download,
            ideally with a proper extension.

            """
            content_type = response.info().get('content-type', '')
            filename = link.filename  # fallback
            # Have a look at the Content-Disposition header for a better guess:
            content_disposition = response.info().get('content-disposition')
            if content_disposition:
                type, params = cgi.parse_header(content_disposition)
                # We use ``or`` here because we don't want to use an "empty" value
                # from the filename param:
                filename = params.get('filename') or filename
            ext = splitext(filename)[1]
            if not ext:
                ext = mimetypes.guess_extension(content_type)
                if ext:
                    filename += ext
            if not ext and link.url != response.geturl():
                ext = splitext(response.geturl())[1]
                if ext:
                    filename += ext
            return filename

        # Descended from _download_url() in pip 1.4.1
        def pipe_to_file(response, path, size=0):
            """Pull the data off an HTTP response, shove it in a new file, and
            show progress.

            :arg response: A file-like object to read from
            :arg path: The path of the new file
            :arg size: The expected size, in bytes, of the download. 0 for
                unknown or to suppress progress indication (as for cached
                downloads)

            """
            def response_chunks(chunk_size):
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

            print('Downloading %s%s...' % (
                self._req.req,
                (' (%sK)' % (size / 1000)) if size > 1000 else ''))
            progress_indicator = (DownloadProgressBar(max=size).iter if size
                                  else DownloadProgressSpinner().iter)
            with open(path, 'wb') as file:
                for chunk in progress_indicator(response_chunks(4096), 4096):
                    file.write(chunk)

        url = link.url.split('#', 1)[0]
        try:
            response = opener(urlparse(url).scheme != 'http').open(url)
        except (HTTPError, IOError) as exc:
            raise DownloadError(link, exc)
        filename = best_filename(link, response)
        try:
            size = int(response.headers['content-length'])
        except (ValueError, KeyError, TypeError):
            size = 0
        pipe_to_file(response, join(self._temp_path, filename), size=size)
        return filename