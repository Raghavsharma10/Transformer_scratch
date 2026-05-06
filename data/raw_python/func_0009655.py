def unblock(self, other_user_id):
        """Unblock the given user.

        :param str other_user_id: the ID of the user to unblock
        :return: ``True`` if successful
        :rtype: bool
        """
        params = {'user': self.user_id, 'otherUser': other_user_id}
        response = self.session.delete(self.url, params=params)
        return response.ok