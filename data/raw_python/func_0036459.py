def rename_group(self, group, newname):
        """Rename contact group

        :param group: group data or name of group
        :param newname: of group
        :type group: ``str``, ``unicode``, ``dict``
        :type newname: ``str``, ``unicode``
        :rtype: ``bool``
        """

        if isinstance(group, basestring):
            group = self.get_contact(group)

        method, url = get_URL('group_update')

        payload = {
            'apikey': self.config.get('apikey'),
            'logintoken': self.session.cookies.get('logintoken'),
            'contactgroupid': group['contactgroupid'],
            'name': newname
            }

        res = getattr(self.session, method)(url, params=payload)

        if res.status_code == 200:
            return True

        hellraiser(res)