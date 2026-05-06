def download(self, path=''):
        """Download the data for this asset.

        :param path: (optional), path where the file should be saved
            to, default is the filename provided in the headers and will be
            written in the current directory.
            it can take a file-like object as well
        :type path: str, file
        :returns: bool -- True if successful, False otherwise
        """
        headers = {
            'Accept': 'application/octet-stream'
            }
        resp = self._get(self._api, allow_redirects=False, stream=True,
                         headers=headers)
        if resp.status_code == 302:
            # Amazon S3 will reject the redirected request unless we omit
            # certain request headers
            headers.update({
                'Content-Type': None,
                })
            with self._session.no_auth():
                resp = self._get(resp.headers['location'], stream=True,
                                 headers=headers)

        if self._boolean(resp, 200, 404):
            stream_response_to_file(resp, path)
            return True
        return False