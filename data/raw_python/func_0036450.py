def get_sent(self, expired=False, for_all=False):
        """Retreve information on previously sent transfers.

        :param expired: Whether or not to return expired transfers.
        :param for_all: Get transfers for all users.
         Requires a Filemail Business account.
        :type for_all: bool
        :type expired: bool
        :rtype: ``list`` of :class:`pyfilemail.Transfer` objects
        """

        method, url = get_URL('get_sent')

        payload = {
            'apikey': self.session.cookies.get('apikey'),
            'logintoken': self.session.cookies.get('logintoken'),
            'getexpired': expired,
            'getforallusers': for_all
            }

        res = getattr(self.session, method)(url, params=payload)

        if res.status_code == 200:
            return self._restore_transfers(res)

        hellraiser(res.json())