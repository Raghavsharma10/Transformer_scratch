def update_membership(self, nickname=None, **kwargs):
        """Update your own membership.

        Note that this fails on former groups.

        :param str nickname: new nickname
        :return: updated membership
        :rtype: :class:`~groupy.api.members.Member`
        """
        return self.memberships.update(nickname=nickname, **kwargs)