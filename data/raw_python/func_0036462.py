def update_company(self, company):
        """Update company settings

        :param company: updated settings
        :type company: ``dict``
        :rtype: ``bool``
        """

        if not isinstance(company, dict):
            raise AttributeError('company must be a <dict>')

        method, url = get_URL('company_update')

        payload = {
            'apikey': self.config.get('apikey'),
            'logintoken': self.session.cookies.get('logintoken')
            }

        payload.update(company)

        res = getattr(self.session, method)(url, params=payload)

        if res.status_code == 200:
            return True

        hellraiser(res)