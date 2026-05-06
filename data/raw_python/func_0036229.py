def _initialize(self):
        """Initialize transfer."""

        payload = {
            'apikey': self.session.cookies.get('apikey'),
            'source': self.session.cookies.get('source')
            }

        if self.fm_user.logged_in:
            payload['logintoken'] = self.session.cookies.get('logintoken')

        payload.update(self.transfer_info)

        method, url = get_URL('init')

        res = getattr(self.session, method)(url, params=payload)

        if res.status_code == 200:
            for key in ['transferid', 'transferkey', 'transferurl']:
                self.transfer_info[key] = res.json().get(key)

        else:
            hellraiser(res)