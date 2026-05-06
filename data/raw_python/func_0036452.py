def update_user_info(self, **kwargs):
        """Update user info and settings.

        :param \*\*kwargs: settings to be merged with
         :func:`User.get_configfile` setings and sent to Filemail.
        :rtype: ``bool``
        """

        if kwargs:
            self.config.update(kwargs)

        method, url = get_URL('user_update')

        res = getattr(self.session, method)(url, params=self.config)

        if res.status_code == 200:
            return True

        hellraiser(res)