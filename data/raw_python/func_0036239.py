def cancel(self):
        """Cancel the current transfer.

        :rtype: ``bool``
        """

        method, url = get_URL('cancel')

        payload = {
            'apikey': self.config.get('apikey'),
            'transferid': self.transfer_id,
            'transferkey': self.transfer_info.get('transferkey')
            }

        res = getattr(self.session, method)(url, params=payload)

        if res.status_code == 200:
            self._complete = True
            return True

        hellraiser(res)