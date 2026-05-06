def get_user_info(self, save_to_config=True):
        """Get user info and settings from Filemail.

        :param save_to_config: Whether or not to save settings to config file
        :type save_to_config: ``bool``
        :rtype: ``dict`` containig user information and default settings.
        """

        method, url = get_URL('user_get')

        payload = {
            'apikey': self.config.get('apikey'),
            'logintoken': self.session.cookies.get('logintoken')
            }

        res = getattr(self.session, method)(url, params=payload)

        if res.status_code == 200:
            settings = res.json()['user']

            if save_to_config:
                self.config.update(settings)

            return settings

        hellraiser(res)