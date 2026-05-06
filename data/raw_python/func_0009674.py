def get_new_members(self, results):
        """Return the newly added members.

        :param results: the results of a membership request check
        :type results: :class:`list`
        :return: the successful requests, as :class:`~groupy.api.memberships.Members`
        :rtype: generator
        """
        for member in results:
            guid = member.pop('guid')
            yield Member(self.manager, self.group_id, **member)
            member['guid'] = guid