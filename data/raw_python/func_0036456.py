def update_contact(self, contact):
        """Update name and/or email for contact.

        :param contact: with updated info
        :type contact: ``dict``
        :rtype: ``bool``
        """

        if not isinstance(contact, dict):
            raise AttributeError('contact must be a <dict>')

        method, url = get_URL('contacts_update')

        payload = {
            'apikey': self.config.get('apikey'),
            'logintoken': self.session.cookies.get('logintoken'),
            'contactid': contact.get('contactid'),
            'name': contact.get('name'),
            'email': contact.get('email')
            }

        res = getattr(self.session, method)(url, params=payload)

        if res.status_code == 200:
            return True

        hellraiser(res)