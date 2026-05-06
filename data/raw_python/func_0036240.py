def rename_file(self, fmfile, newname):
        """Rename file in transfer.

        :param fmfile: file data from filemail containing fileid
        :param newname: new file name
        :type fmfile: ``dict``
        :type newname: ``str`` or ``unicode``
        :rtype: ``bool``
        """

        if not isinstance(fmfile, dict):
            raise FMBaseError('fmfile must be a <dict>')

        method, url = get_URL('file_rename')

        payload = {
            'apikey': self.config.get('apikey'),
            'logintoken': self.session.cookies.get('logintoken'),
            'fileid': fmfile.get('fileid'),
            'filename': newname
            }

        res = getattr(self.session, method)(url, params=payload)
        if res.status_code == 200:
            self._complete = True
            return True

        hellraiser(res)