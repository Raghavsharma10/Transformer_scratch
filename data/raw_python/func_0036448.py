def logout(self):
        """Logout of filemail and closing the session."""

        # Check if all transfers are complete before logout
        self.transfers_complete

        payload = {
            'apikey': self.config.get('apikey'),
            'logintoken': self.session.cookies.get('logintoken')
            }

        method, url = get_URL('logout')
        res = getattr(self.session, method)(url, params=payload)

        if res.status_code == 200:
            self.session.cookies['logintoken'] = None
            return True

        hellraiser(res)