def compress(self):
        """Compress files on the server side after transfer complete
         and make zip available for download.

        :rtype: ``bool``
        """

        method, url = get_URL('compress')

        payload = {
            'apikey': self.config.get('apikey'),
            'logintoken': self.session.cookies.get('logintoken'),
            'transferid': self.transfer_id
            }

        res = getattr(self.session, method)(url, params=payload)

        if res.status_code == 200:
            return True

        hellraiser(res)