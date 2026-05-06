def archive(self, format, path='', ref='master'):
        """Get the tarball or zipball archive for this repo at ref.

        See: http://developer.github.com/v3/repos/contents/#get-archive-link

        :param str format: (required), accepted values: ('tarball',
            'zipball')
        :param path: (optional), path where the file should be saved
            to, default is the filename provided in the headers and will be
            written in the current directory.
            it can take a file-like object as well
        :type path: str, file
        :param str ref: (optional)
        :returns: bool -- True if successful, False otherwise

        """
        resp = None
        if format in ('tarball', 'zipball'):
            url = self._build_url(format, ref, base_url=self._api)
            resp = self._get(url, allow_redirects=True, stream=True)

        if resp and self._boolean(resp, 200, 404):
            stream_response_to_file(resp, path)
            return True
        return False