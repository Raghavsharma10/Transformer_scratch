def get_files(self):
        """Get information on file in transfer from Filemail.

        :rtype: ``list`` of ``dict`` objects with info on files
        """

        method, url = get_URL('get')
        payload = {
            'apikey': self.session.cookies.get('apikey'),
            'logintoken': self.session.cookies.get('logintoken'),
            'transferid': self.transfer_id,
            }

        res = getattr(self.session, method)(url, params=payload)

        if res.status_code == 200:
            transfer_data = res.json()['transfer']
            files = transfer_data['files']

            for file_data in files:
                self._files.append(file_data)

            return self.files

        hellraiser(res)