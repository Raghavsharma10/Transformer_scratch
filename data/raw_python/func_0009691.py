def get_membership(self):
        """Get your membership.

        Note that your membership may not exist. For example, you do not have
        a membership in a former group. Also, the group returned by the API
        when rejoining a former group does not contain your membership. You
        must call :func:`refresh_from_server` to update the list of members.

        :return: your membership in the group
        :rtype: :class:`~groupy.api.memberships.Member`
        :raises groupy.exceptions.MissingMembershipError: if your membership is
                not in the group data
        """
        user_id = self._user.me['user_id']
        for member in self.members:
            if member.user_id == user_id:
                return member
        raise exceptions.MissingMembershipError(self.group_id, user_id)