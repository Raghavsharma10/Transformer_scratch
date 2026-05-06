def delete_group(self, name):
        """Delete contact group

        :param name: of group
        :type name: ``str``, ``unicode``
        :rtype: ``bool``
        """

        group = self.get_group(name)

        method, url = get_URL('group_delete')

        payload = {
            'apikey': self.config.get('apikey'),
            'logintoken': self.session.cookies.get('logintoken'),
            'contactgroupid': group['contactgroupid']
            }

        res = getattr(self.session, method)(url, params=payload)

        if res.status_code == 200:
            return True

        hellraiser(res)