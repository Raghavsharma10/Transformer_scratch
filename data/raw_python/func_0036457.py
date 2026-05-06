def get_group(self, name):
        """Get contact group by name

        :param name: name of group
        :type name: ``str``, ``unicode``
        :rtype: ``dict`` with group data
        """

        groups = self.get_groups()
        for group in groups:
            if group['contactgroupname'] == name:
                return group

        msg = 'No group named: "{name}" found.'
        raise FMBaseError(msg.format(name=name))