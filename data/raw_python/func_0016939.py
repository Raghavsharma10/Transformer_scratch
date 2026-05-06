def edit(self,
             billing_email=None,
             company=None,
             email=None,
             location=None,
             name=None):
        """Edit this organization.

        :param str billing_email: (optional) Billing email address (private)
        :param str company: (optional)
        :param str email: (optional) Public email address
        :param str location: (optional)
        :param str name: (optional)
        :returns: bool
        """
        json = None
        data = {'billing_email': billing_email, 'company': company,
                'email': email, 'location': location, 'name': name}
        self._remove_none(data)

        if data:
            json = self._json(self._patch(self._api, data=dumps(data)), 200)

        if json:
            self._update_(json)
            return True
        return False