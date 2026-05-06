def add(self, nickname, email=None, phone_number=None, user_id=None):
        """Add a user to the group.

        You must provide either the email, phone number, or user_id that
        uniquely identifies a user.

        :param str nickname: new name for the user in the group
        :param str email: email address of the user
        :param str phone_number: phone number of the user
        :param str user_id: user_id of the user
        :return: a membership request
        :rtype: :class:`MembershipRequest`
        """
        member = {
            'nickname': nickname,
            'email': email,
            'phone_number': phone_number,
            'user_id': user_id,
        }
        return self.add_multiple(member)