def add_to_group(self, group_id, nickname=None):
        """Add the member to another group.

        If a nickname is not provided the member's current nickname is used.

        :param str group_id: the group_id of a group
        :param str nickname: a new nickname
        :return: a membership request
        :rtype: :class:`MembershipRequest`
        """
        if nickname is None:
            nickname = self.nickname
        memberships = Memberships(self.manager.session, group_id=group_id)
        return memberships.add(nickname, user_id=self.user_id)