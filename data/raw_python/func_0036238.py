def share(self, to, sender=None, message=None):
        """Share transfer with new message to new people.

        :param to: receiver(s)
        :param sender: Alternate email address as sender
        :param message: Meggase to new recipients
        :type to: ``list`` or ``str`` or ``unicode``
        :type sender: ``str`` or ``unicode``
        :type message: ``str`` or ``unicode``
        :rtyep: ``bool``
        """

        method, url = get_URL('share')

        payload = {
            'apikey': self.session.cookies.get('apikey'),
            'logintoken': self.session.cookies.get('logintoken'),
            'transferid': self.transfer_id,
            'to': self._parse_recipients(to),
            'from': sender or self.fm_user.username,
            'message': message or ''
            }

        res = getattr(self.session, method)(url, params=payload)

        if res.status_code == 200:
            return True

        hellraiser(res)