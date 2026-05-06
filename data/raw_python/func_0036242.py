def update(self,
               message=None,
               subject=None,
               days=None,
               downloads=None,
               notify=None):
        """Update properties for a transfer.

        :param message: updated message to recipient(s)
        :param subject: updated subject for trasfer
        :param days: updated amount of days transfer is available
        :param downloads: update amount of downloads allowed for transfer
        :param notify: update whether to notifiy on downloads or not
        :type message: ``str`` or ``unicode``
        :type subject: ``str`` or ``unicode``
        :type days: ``int``
        :type downloads: ``int``
        :type notify: ``bool``
        :rtype: ``bool``
        """

        method, url = get_URL('update')

        payload = {
            'apikey': self.config.get('apikey'),
            'logintoken': self.session.cookies.get('logintoken'),
            'transferid': self.transfer_id,
            }

        data = {
            'message': message or self.transfer_info.get('message'),
            'message': subject or self.transfer_info.get('subject'),
            'days': days or self.transfer_info.get('days'),
            'downloads': downloads or self.transfer_info.get('downloads'),
            'notify': notify or self.transfer_info.get('notify')
            }

        payload.update(data)

        res = getattr(self.session, method)(url, params=payload)

        if res.status_code:
            self.transfer_info.update(data)
            return True

        hellraiser(res)