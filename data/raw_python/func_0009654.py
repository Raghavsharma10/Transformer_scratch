def block(self, other_user_id):
        """Block the given user.

        :param str other_user_id: the ID of the user to block
        :return: the block created
        :rtype: :class:`~groupy.api.blocks.Block`
        """
        params = {'user': self.user_id, 'otherUser': other_user_id}
        response = self.session.post(self.url, params=params)
        block = response.data['block']
        return Block(self, **block)