def between(self, other_user_id):
        """Check if there is a block between you and the given user.

        :return: ``True`` if the given user has been blocked
        :rtype: bool
        """
        params = {'user': self.user_id, 'otherUser': other_user_id}
        response = self.session.get(self.url, params=params)
        return response.data['between']