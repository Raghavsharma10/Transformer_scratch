def add_contact_to_group(self, contact, group):
        """Add contact to group

        :param contact: name or contact object
        :param group: name or group object
        :type contact: ``str``, ``unicode``, ``dict``
        :type group: ``str``, ``unicode``, ``dict``
        :rtype: ``bool``
        """

        if isinstance(contact, basestring):
            contact = self.get_contact(contact)

        if isinstance(group, basestring):
            group = self.get_group(group)

        method, url = get_URL('contacts_add_to_group')

        payload = {
            'apikey': self.config.get('apikey'),
            'logintoken': self.session.cookies.get('logintoken'),
            'contactid': contact['contactid'],
            'contactgroupid': group['contactgroupid']
            }

        res = getattr(self.session, method)(url, params=payload)

        if res.status_code == 200:
            return True

        hellraiser(res)