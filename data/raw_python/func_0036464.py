def company_add_user(self, email, name, password, receiver, admin):
        """Add a user to the company account.

        :param email:
        :param name:
        :param password: Pass without storing in plain text
        :param receiver: Can user receive files
        :param admin:
        :type email: ``str`` or ``unicode``
        :type name: ``str`` or ``unicode``
        :type password: ``str`` or ``unicode``
        :type receiver: ``bool``
        :type admin: ``bool``
        :rtype: ``bool``
        """

        method, url = get_URL('company_add_user')

        payload = {
            'apikey': self.config.get('apikey'),
            'logintoken': self.session.cookies.get('logintoken'),
            'email': email,
            'name': name,
            'password': password,
            'canreceivefiles': receiver,
            'admin': admin
            }

        res = getattr(self.session, method)(url, params=payload)

        if res.status_code == 200:
            return True

        hellraiser(res)