def add_multiple(self, *users):
        """Add multiple users to the group at once.

        Each given user must be a dictionary containing a nickname and either
        an email, phone number, or user_id.

        :param args users: the users to add
        :return: a membership request
        :rtype: :class:`MembershipRequest`
        """
        guid = uuid.uuid4()
        for i, user_ in enumerate(users):
            user_['guid'] = '{}-{}'.format(guid, i)

        payload = {'members': users}
        url = utils.urljoin(self.url, 'add')
        response = self.session.post(url, json=payload)
        return MembershipRequest(self, *users, group_id=self.group_id,
                                 **response.data)