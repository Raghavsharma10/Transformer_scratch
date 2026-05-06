def delete_file(self, fmfile):
        """Delete file from transfer.

        :param fmfile: file data from filemail containing fileid
        :type fmfile: ``dict``
        :rtype: ``bool``
        """

        if not isinstance(fmfile, dict):
            raise FMFileError('fmfile must be a <dict>')

        method, url = get_URL('file_delete')

        payload = {
            'apikey': self.config.get('apikey'),
            'logintoken': self.session.cookies.get('logintoken'),
            'fileid': fmfile.get('fileid')
            }

        res = getattr(self.session, method)(url, params=payload)

        if res.status_code == 200:
            self._complete = True
            return True

        hellraiser(res)