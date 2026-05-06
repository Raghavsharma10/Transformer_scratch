def get_memberships(self):
        """Fetches all group memberships.

        Returns:
            dict:
        key: group name
        value: (array of users, array of groups)
        """

        response = self._get_xml(self.rest_url + "/group/membership")

        if not response.ok:
            return None

        xmltree = etree.fromstring(response.content)
        memberships = {}
        for mg in xmltree.findall('membership'):
            # coerce values to unicode in a python 2 and 3 compatible way
            group = u'{}'.format(mg.get('group'))
            users = [u'{}'.format(u.get('name')) for u in mg.find('users').findall('user')]
            groups = [u'{}'.format(g.get('name')) for g in mg.find('groups').findall('group')]
            memberships[group] = {u'users': users, u'groups': groups}
        return memberships