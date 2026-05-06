def list(self):
        """List the users you have blocked.

        :return: a list of :class:`~groupy.api.blocks.Block`'s
        :rtype: :class:`list`
        """
        params = {'user': self.user_id}
        response = self.session.get(self.url, params=params)
        blocks = response.data['blocks']
        return [Block(self, **block) for block in blocks]