def get_company_info(self):
        """Get company settings from Filemail

        :rtype: ``dict`` with company data
        """

        method, url = get_URL('company_get')

        payload = {
            'apikey': self.config.get('apikey'),
            'logintoken': self.session.cookies.get('logintoken')
            }

        res = getattr(self.session, method)(url, params=payload)

        if res.status_code == 200:
            return res.json()['company']

        hellraiser(res)