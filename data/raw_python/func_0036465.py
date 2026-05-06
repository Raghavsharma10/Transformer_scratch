def update_company_user(self, email, userdata):
        """Update a company users settings

        :param email: current email address of user
        :param userdata: updated settings
        :type email: ``str`` or ``unicode``
        :type userdata: ``dict``
        :rtype: ``bool``
        """

        if not isinstance(userdata, dict):
            raise AttributeError('userdata must be a <dict>')

        method, url = get_URL('company_update_user')

        payload = {
            'apikey': self.config.get('apikey'),
            'logintoken': self.session.cookies.get('logintoken'),
            'useremail': email
            }

        payload.update(userdata)

        res = getattr(self.session, method)(url, params=payload)

        if res.status_code == 200:
            return True

        hellraiser(res)