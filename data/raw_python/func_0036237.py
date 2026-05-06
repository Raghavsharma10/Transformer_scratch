def forward(self, to):
        """Forward prior transfer to new recipient(s).

        :param to: new recipients to a previous transfer.
         Use ``list`` or  comma seperatde ``str`` or ``unicode`` list
        :type to: ``list`` or ``str`` or ``unicode``
        :rtype: ``bool``

        """

        method, url = get_URL('forward')

        payload = {
            'apikey': self.session.cookies.get('apikey'),
            'transferid': self.transfer_id,
            'transferkey': self.transfer_info.get('transferkey'),
            'to': self._parse_recipients(to)
            }

        res = getattr(self.session, method)(url, params=payload)

        if res.status_code == 200:
            return True

        hellraiser(res)